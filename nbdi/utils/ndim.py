""" This file defines adapters to generalize across numpy and pytorch

Author: Oleg Rybkin (olehrybkin.com, oleh.rybkin@gmail.com)
"""

import numpy as np
import torch
import sys
import numbers


class adapter:
    """ A function decorator that redirects the function to the appropriate numpy or torch analogue """
    
    def __init__(self, torch_fn=None, numpy_fn=None):
        """ Decorator parameters """
        self.torch_fn = torch_fn
        self.numpy_fn = numpy_fn
    
    def __call__(self, _):
        """ Wrapping """
        
        def wrapper(generalized_tensor, *args, **kwargs):
            found_tensor = generalized_tensor
            if isinstance(generalized_tensor, list):
                found_tensor = generalized_tensor[0]
            
            if isinstance(found_tensor, torch.Tensor):
                fn = self.torch_fn
            elif isinstance(found_tensor, np.ndarray) or isinstance(found_tensor, numbers.Number):
                # numpy is used if the input is a number. this is because numpy functions usually work with numbers
                # whereas torch functions don't
                fn = self.numpy_fn
            else:
                raise TypeError("Do not currently support this data type {}".format(type(generalized_tensor)))

            return fn(generalized_tensor, *args, **kwargs)
        
        return wrapper


class Ndim():
    """ This class is the public interface of the module. This is necessary to override __getattr__.
    See https://stackoverflow.com/questions/2447353/getattr-on-a-module """
    
    @staticmethod
    @adapter(torch.transpose, np.swapaxes)
    def swapaxes(*args, **kwargs):
        pass
    
    @staticmethod
    @adapter(torch.unsqueeze, np.expand_dims)
    def unsqueeze(*args, **kwargs):
        pass
    
    @staticmethod
    @adapter(torch.clone, np.copy)
    def copy(*args, **kwargs):
        pass
    
    @staticmethod
    @adapter(torch.Tensor.type, np.ndarray.astype)
    def astype(*args, **kwargs):
        pass

    @staticmethod
    def cat(arrs, dim):
        if isinstance(arrs[0], torch.Tensor):
            return torch.cat(arrs, dim)
        else:
            return np.concatenate(arrs, dim)

    @staticmethod
    def index_select(arr, dim, index):
        if isinstance(arr, torch.Tensor):
            return torch.index_select(arr, dim, index)
        else:
            return np.take(arr, index, dim)
    
    def __getattr__(self, name):
        return adapter(getattr(torch, name), getattr(np, name))(None)

sys.modules[__name__] = Ndim()
