import torch
import torch.nn as nn
from prim_ops import OPS, DownOps, UpOps, NormOps, ConvOps
# from helper import dim_assert
import pdb


class MixedOp(nn.Module):
    def __init__(self, channels, stride, transposed=False):
        '''
        channels: in_channels == out_channels for MixedOp
        '''
        super().__init__()
        self._ops = nn.ModuleList()
        self.stride = stride
        if stride == 1:
            primitives = NormOps
        else:
            primitives = UpOps if transposed else DownOps
        for pri in primitives:
            op = OPS[pri](channels, stride)
            self._ops.append(op)

    def forward(self, x, alpha1, alpha2):
        '''
        alpha1: Weights for MixedOps with stride=1
        alpha2: Weights for MixedOps with stride=2
        '''
        if self.stride == 1:
            res = sum([w * op(x) for w, op in zip(alpha1, self._ops)]) # debug: dim_assert
        else:
            res = sum([w * op(x) for w, op in zip(alpha2, self._ops)]) # debug: dim_assert
        return res

class Cell(nn.Module):
    def __init__(self, n_nodes, c0, c1, c_node, downward=True):
        '''
        n_nodes: How many nodes in a cell.
        c0, c1: in_channels for two inputs.
        c_node: out_channels for each node.
        downward: If True, this is a downward block, otherwise, an upward block.
        '''
        super().__init__()
        self.n_nodes = n_nodes
        self.c_node = c_node

        self.preprocess0 = ConvOps(c0, c_node, kernel_size=1, 
                                   stride = 2 if downward else 1, 
                                   ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c1, c_node, kernel_size=1, ops_order='act_weight_norm')

        self._ops = nn.ModuleList()
        
        for n_edges in range(2, 2+n_nodes):
            for i in range(n_edges):
                if downward:
                    self._ops.append(MixedOp(c_node, stride = 2 if i <= 1 else 1))
                else:
                    self._ops.append(MixedOp(c_node, stride = 2 if i == 1 else 1, transposed = True))
        return

    @property
    def out_channels(self):
        return self.n_nodes * self.c_node

    def forward(self, x0, x1, alpha1, alpha2):
        '''
        x0, x1: Inputs of cell
        alpha1: Weights for MixedOp with stride == 1
        alpha2: Weights for MixedOp with stride == 2
        '''
        x0 = self.preprocess0(x0)
        x1 = self.preprocess1(x1)
        xs = [x0, x1]
        i = 0
        for node in range(self.n_nodes):
            outputs = []
            for x in xs:
                outputs.append(self._ops[i](x, alpha1[i], alpha2[i]))
                i += 1
            xs.append(sum(outputs)) # debug: dim_assert
        return torch.cat(xs[-self.n_nodes:], dim=1) # debug: dim_assert
            
       