from torch.functional import F
import torch.nn as nn
from prim_ops import OPS, DownOps, UpOps, NormOps, ConvOps
from helper import consistent_dim
import pdb


class MixedOp(nn.Module):
    def __init__(self, channels, stride, transposed=False):
        '''
        channels: in_channels == out_channels for MixedOp
        '''
        super().__init__()
        self._ops = nn.ModuleList()
        if stride >= 2: # down or up edge
            primitives = UpOps if transposed else DownOps
            self._op_type = 'up_or_down'
        else:
            primitives = NormOps
            self._op_type = 'normal'
        for pri in primitives:
            op = OPS[pri](channels, stride)
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
    def __init__(self, n_nodes, c0, c1, c_node, cell_type):
        '''
        n_nodes: How many nodes in a cell.
        c0, c1: in_channels for two inputs.
        c_node: out_channels for each node.
        cell_type: 'up' or 'down'
        '''
        super().__init__()
        self.n_nodes = n_nodes
        self.c_node = c_node

        self.preprocess0 = ConvOps(c0, c_node, kernel_size=1, 
                                   stride = 2 if cell_type == 'down' else 1, 
                                   ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c1, c_node, kernel_size=1, ops_order='act_weight_norm')

        self._ops = nn.ModuleList()
        
        for n_edges in range(2, 2+n_nodes):
            for i in range(n_edges):
                if cell_type == 'down':
                    self._ops.append(MixedOp(c_node, stride = 2 if i <= 1 else 1))
                else:
                    self._ops.append(MixedOp(c_node, stride = 2 if i == 1 else 1, transposed = True))
        return

    @property
    def out_channels(self):
        return self.n_nodes * self.c_node

    def forward(self, input0, input1, w1, w2):
        '''
        input0, input1: inputs of cell
        w1: weights for MixedOp with stride == 1
        w2: weights for MixedOp with stride == 2
        '''
        input0 = self.preprocess0(input0)
        input1 = self.preprocess1(input1)
        inputs = [input0, input1]
        i_w = 0
        for node in range(self.n_nodes):
            outputs = []
            for i, input_i in enumerate(inputs):
                i_w += i
                outputs.append(self._ops[i_w](input_i, w1[i_w], w2[i_w]))
            inputs.append(sum(consistent_dim(outputs)))
        return torch.cat(inputs[-self.n_nodes:], dim=1)
            
       