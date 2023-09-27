import os

from nbdi.models.closed_loop_nbdi_mdl import ClNBDIRLMdl
from nbdi.components.logger import Logger
from nbdi.utils.general_utils import AttrDict
from nbdi.configs.default_data_configs.kitchen import data_spec
from nbdi.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ClNBDIRLMdl,
    'logger': Logger,
    'data_dir': '.',
    'epoch_cycles_train': 50,
    'num_epochs': 100,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=30,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
