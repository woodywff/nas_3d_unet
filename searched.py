import torch
import torch.nn as nn
from prim_ops import OPS, ConvOps
# from helper import dim_assert
import pdb
from genotype import Genotype

FLAG_DEBUG = True

class SearchedCell(nn.Module):
    def __init__(self, n_nodes, c0, c1, c_node, gene, downward=True):
        '''
        n_nodes: How many nodes in a cell.
        c0, c1: in_channels for two inputs.
        c_node: out_channels for each node.
        gene: Genotype, searched architecture of a cell
        downward: If True, this is a downward block, otherwise, an upward block.
        '''
        super().__init__()
        self.n_nodes = n_nodes
        self.c_node = c_node
        self.genolist = gene.down if downward else gene.up
        
        self.preprocess0 = ConvOps(c0, c_node, kernel_size=1, 
                                   stride = 2 if downward else 1, 
                                   ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c1, c_node, kernel_size=1, ops_order='act_weight_norm')
        
        self._ops = nn.ModuleList([OPS[i[0]](c_node) for i in self.genolist])
        
        return

    @property
    def out_channels(self):
        return self.n_nodes * self.c_node

    def forward(self, x0, x1):
        '''
        x0, x1: Inputs to a cell
        '''
        x0 = self.preprocess0(x0)
        x1 = self.preprocess1(x1)
        xs = [x0, x1]
        i = 0
        for node in range(self.n_nodes):
            outputs = []
            for _ in range(2):
                outputs.append(self._ops[i](xs[self.genolist[i][1]]))
                i += 1
            xs.append(sum(outputs)) # debug: dim_assert
        return torch.cat(xs[-self.n_nodes:], dim=1) # debug: dim_assert
            

class SearchedNet(nn.Module):
    def __init__(self, in_channels, init_n_kernels, out_channels, depth, n_nodes, channel_change,
                 gene):
        '''
        This class defines the U-shaped architecture. I take it as the kernel of NAS. 
        in_channels: How many kinds of MRI modalities being used.
        init_n_kernels: Number of kernels for the nodes in the first cell.
        out_channels: How many kinds of tumor labels.
        depth: Number of downward cells. For upward, it has depth+1 cells.
        n_nodes: Number of nodes in each cell.
        channel_change: If True, channel size expands and shrinks in double during downward and upward forwarding.
        gene: searched cell.
        '''
        super().__init__()
        c0 = c1 = n_nodes * init_n_kernels # channel0, channel1, the number of kernels.
        c_node = init_n_kernels 

        self.stem0 = ConvOps(in_channels, c0, kernel_size=1, ops_order='weight_norm')
        self.stem1 = ConvOps(in_channels, c1, kernel_size=3,  stride=2, ops_order='weight_norm')

        self.down_cells = nn.ModuleList()
        self.up_cells = nn.ModuleList()

        down_channels = [c0, c1]
        for i in range(depth):
            c_node = 2 * c_node if channel_change else c_node  # double the number of filters
            down_cell = SearchedCell(n_nodes, c0, c1, c_node, gene)
            self.down_cells += [down_cell]
            c0, c1 = c1, down_cell.out_channels
            down_channels.append(c1)
        down_channels.pop()
        for i in range(depth+1):
            c0 = down_channels.pop()
            up_cell = SearchedCell(n_nodes, c0, c1, c_node, gene, downward = False)
            self.up_cells += [up_cell]
            c1 = up_cell.out_channels
            c_node = c_node // 2 if channel_change else c_node  # halve the number of filters
        self.last_conv = nn.Sequential(ConvOps(c1, out_channels, kernel_size=1, 
                                               dropout_rate=0.5, ops_order='weight'),# dropout_rate is different for searching and training
                                       nn.Sigmoid())

    def forward(self, x):
        s0, s1 = self.stem0(x), self.stem1(x)
        down_outputs = [s0, s1]
        for i, cell in enumerate(self.down_cells):
            s0, s1 = s1, cell(s0, s1)
            down_outputs.append(s1)
        if FLAG_DEBUG:
            print('x.shape = ',x.shape)
            for i in down_outputs: 
                print(i.shape)
        down_outputs.pop()
        for i, cell in enumerate(self.up_cells):
            s0 = down_outputs.pop()
            s1 = cell(s0, s1)
            if FLAG_DEBUG:
                print(s1.shape)
        return self.last_conv(s1)
         