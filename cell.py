from torch.functional import F
import torch.nn as nn
from prim_ops import OPS, DownOps, UpOps, NormOps
# from util.utils import consistent_dim
import pdb


class MixedOp(nn.Module):

    def __init__(self, c_node, stride, use_transpose=False):
        '''
        c_node: in_channels == out_channels for MixedOp
        '''
        super().__init__()
        self._ops = nn.ModuleList()
        if stride >= 2: # down or up edge
            primitives = UpOps if use_transpose else DownOps
            self._op_type = 'up_or_down'
        else:
            primitives = NormOps
            self._op_type = 'normal'
        for pri in primitives:
            op = OPS[pri](c_node, stride)
            self._ops.append(op)

    def forward(self, x, w1, w2):
        '''
        w1: normal, stride=1
        w2: up_or_down, stride=2
        '''

        if self._op_type == 'up_or_down':
            res = sum(w * op(x) for w, op in zip(w2, self._ops))
        else:
            res = sum(w * op(x) for w, op in zip(w1, self._ops))

        return res

class Cell(nn.Module):

    def __init__(self, n_nodes, c_prev_prev, c_prev, c_node, cell_type):
        
        super().__init__()
        self.c_prev_prev = c_prev_prev
        self.c_prev = c_prev
        self.c_node = c_node
        self._n_nodes = n_nodes
        self._multiplier = n_nodes
        self._input_node_num = 2
        self.out_channels = n_nodes * c_node
        
        return

        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = ConvOps(c_prev_prev, c_node, kernel_size=1, stride=2, affine=False, ops_order='act_weight_norm')
        else:
            self.preprocess0 = ConvOps(c_prev_prev, c_node, kernel_size=1, affine=False, ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c_prev, c_node, kernel_size=1, affine=False, ops_order='act_weight_norm')

        self._ops = nn.ModuleList()

        # inp2changedim = 2 if cell_type == 'down' else 1
        idx_up_or_down_start = 0 if cell_type == 'down' else 1
        for i in range(self._n_nodes):
            for j in range(self._input_node_num + i): # the input id for remaining meta-node
                stride = 2 if j < 2 and j >= idx_up_or_down_start else 1
                op = MixedOp(c_node, stride, use_transpose=True) if cell_type=='up' else MixedOp(c_node, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weight1, weight2):
        # weight1: the normal operations weights with sharing
        # weight2: the down or up operations weight, respectively

        # the cell output is concatenate, so need a convolution to learn best combination
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0

        for i in range(self._n_nodes):
            # handle the un-consistent dimension
            tmp_list = []
            for j, h in enumerate(states):
                tmp_list += [self._ops[offset+j](h, weight1[offset+j], weight2[offset+j])]
            s = sum(consistent_dim(tmp_list))######################################## !!!
            #s = sum(self._ops[offset+j](h, weight1[offset+j], weight2[offset+j]) for j, h in enumerate(states))
            offset += len(states)        
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)
