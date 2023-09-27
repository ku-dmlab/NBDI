import d4rl
import gym
import numpy as np
import itertools

from nbdi.components.data_loader import Dataset
from nbdi.utils.general_utils import AttrDict
import torch

from nbdi.utils.general_utils import AttrDict, map_dict
from novelty_module_kitchen import Phi, Fnet
from torch import nn
import torch
import torch.nn.functional as F


class D4RLSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle

        env = gym.make(self.spec.env_name)
        self.dataset = env.get_dataset()

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            if not isinstance(self.spec.filter_indices[0], list):
                self.spec.filter_indices = [self.spec.filter_indices]
            self.seqs = list(itertools.chain.from_iterable([\
                list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                               for x in self.seqs[fi[0] : fi[1]+1])) for fi in self.spec.filter_indices]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs
            
        self.phi_model = Phi().eval()
        self.phi_model.load_state_dict(torch.load("icm/saved_encoder"))

        self.f_model = Fnet().eval()
        self.f_model.load_state_dict(torch.load("icm/saved_forward"))
        
        self.forward_loss = nn.MSELoss(reduction='none')
        self.inverse_loss = nn.MSELoss(reduction='none')

    def __getitem__(self, index):  
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        skill_length = self.get_skill_length(start_idx, seq)
        
        output = AttrDict(
            states=seq.states[start_idx:start_idx+skill_length+1],
            actions=seq.actions[start_idx:start_idx+skill_length],
            pad_mask=np.ones((skill_length,)),
        )

        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
    
        end_token = np.zeros((output.actions.shape[0], 1)).astype('float32')
        end_token.fill(-1)
        output.actions = np.concatenate((output.actions, end_token), axis=1)

        if output.actions.shape[0] < self.spec.max_seq_len: 
            output.actions[-1][-1] = 1
  
        # Make length consistent
        end_ind = np.argmax(output.pad_mask * np.arange(output.pad_mask.shape[0], dtype=np.float32), 0)
        end_ind, output = self._sample_max_len_video(output, end_ind, target_len=self.spec.max_seq_len+1) 

        mask_seq = np.zeros((self.spec.max_seq_len, 10))
        mask_seq[:skill_length] = [1,1,1,1,1,1,1,1,1,1]
        output.action_mask = mask_seq
    
        return output

    def _sample_seq(self):
        # randomly choose one epsidoe from train data
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)
    
    def _sample_max_len_video(self, data_dict, end_ind, target_len):
        """ This function processes data tensors so as to have length equal to target_len
        by sampling / padding if necessary """
        extra_length = (end_ind + 1) - target_len
        if self.phase == 'train':
            offset = max(0, int(np.random.rand() * (extra_length + 1)))
        else:
            offset = 0

        data_dict = map_dict(lambda tensor: self._maybe_pad(tensor, offset, target_len), data_dict)
        if 'actions' in data_dict:
            data_dict.actions = data_dict.actions[:-1]
        end_ind = min(end_ind - offset, target_len - 1)

        return end_ind, data_dict

    @staticmethod
    def _maybe_pad(val, offset, target_length):
        """Pads / crops sequence to desired length."""
        val = val[offset:]
        len = val.shape[0]
        if len > target_length:
            return val[:target_length]
        elif len < target_length:
            return np.concatenate((val, np.zeros([int(target_length - len)] + list(val.shape[1:]), dtype=val.dtype)))
        else:
            return val
        
    def state_action_novelty_module(self, state1, state2, action, forward_scale=1.0):
        """We use ICM as the state-action novelty moduel"""
        state1_hat = self.phi_model(state1)
        state2_hat = self.phi_model(state2)
        state2_hat_pred = self.f_model(state1_hat.detach(), action.detach())

        forward_pred_err = forward_scale * self.forward_loss(state2_hat_pred, \
                            state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
        
        return forward_pred_err
    
    def get_skill_length(self, start_idx, data, threshold=0.3):
        skill_length = 0
        index = start_idx
        i_reward = 0
        
        while skill_length <= self.spec.max_seq_len and i_reward <= threshold: 
            state1_batch = torch.from_numpy(data.states[index]).unsqueeze(0)
            state2_batch = torch.from_numpy(data.states[index+1]).unsqueeze(0)
            action_batch = torch.from_numpy(data.actions[index]).unsqueeze(0)
            
            with torch.no_grad():
                i_reward = self.state_action_novelty_module(state1_batch, state2_batch, action_batch)

            skill_length += 1
            index += 1

        if skill_length > self.spec.max_seq_len:
            skill_length = self.spec.max_seq_len
            return skill_length
        elif skill_length - 1 == 0:
            return skill_length
        else:
            return skill_length - 1