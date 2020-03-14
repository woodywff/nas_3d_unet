# import subprocess
import numpy as np
from torch.nn.functional import interpolate
import pdb

# def get_gpus_memory_info():
#     '''
#     Return the GPU with the largest free memory.
#     '''
#     rst = subprocess.run('nvidia-smi -q -d Memory',stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
#     rst = rst.strip().split('\n')
#     memory_available = [int(line.split(':')[1].split(' ')[1]) for line in rst if 'Free' in line][::2]
#     max_id = int(np.argmax(memory_available))
#     return max_id, memory_available

def calc_param_size(model):
    return np.sum(np.prod(v.size()) for v in model.parameters())*4e-6

def consistent_dim(tensor_list):
    pdb.set_trace()
    shape = tensor_list[0].size()
    for t in tensor_list:
        assert shape == t.size(), 'inconsistent dim for Add!'
        
    
    dims = tuple(np.max([tensor.size() for tensor in tensor_list], axis=0)[-3:])
    return [interpolate(tensor, dim) for tensor in tensor_list]