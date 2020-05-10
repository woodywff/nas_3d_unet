import numpy as np
from torch.nn.functional import interpolate
import pdb

def minmax_normalize(img_npy):
    '''
    img_npy: ndarray
    '''
    min_value = np.min(img_npy)
    max_value = np.max(img_npy)
    return (img_npy - min_value)/(max_value - min_value)

def calc_param_size(model):
    '''
    Show the memory cost of model.parameters, in MB. 
    '''
    return np.sum(np.prod(v.size()) for v in model.parameters())*4e-6

def dim_assert(t_list):
    '''
    To make sure that all the tensors in t_list has the same dims.
    '''
    dims = tuple(np.max([t.size() for t in t_list], axis=0)[-3:])
    for i in range(len(t_list)):
        if tuple(t_list[i].shape[-3:]) != dims:
            print_red('inconsistent dim: i')
            t_list[i] = interpolate(t_list[i], dims)
    return t_list


def print_red(something):
    print("\033[1;31m{}\033[0m".format(something))
    
