import torch
import torch.nn as nn
from prim_ops import ConvOps, DownOps, UpOps, NormOps
from cell import Cell
from torch.functional import F
import pdb
from genotype import Genotype, GenoParser



FLAG_DEBUG = False

class KernelNet(nn.Module):
    def __init__(self, in_channels, init_n_kernels, out_channels, depth, n_nodes, channel_change):
        '''
        This class defines the U-shaped architecture. I take it as the kernel of NAS. 
        in_channels: How many kinds of MRI modalities being used.
        init_n_kernels: Number of kernels for the nodes in the first cell.
        out_channels: How many kinds of tumor labels.
        depth: Number of downward cells. For upward, it has depth+1 cells.
        n_nodes: Number of nodes in each cell.
        channel_change: If True, channel size expands and shrinks in double during downward and upward forwarding.  
        '''
        super().__init__()
        c0 = c1 = n_nodes * init_n_kernels # channel0, channel1, the number of kernels.
        c_node = init_n_kernels 

        self.stem0 = ConvOps(in_channels, c0, kernel_size=1, ops_order='weight_norm')
        self.stem1 = ConvOps(in_channels, c1, kernel_size=3,  stride=2, ops_order='weight_norm')

        assert depth >= 2 , 'depth must >= 2'

        self.down_cells = nn.ModuleList()
        self.up_cells = nn.ModuleList()

        down_channels = [c0, c1]
        for i in range(depth):
            c_node = 2 * c_node if channel_change else c_node  # double the number of filters
            down_cell = Cell(n_nodes, c0, c1, c_node)
            self.down_cells += [down_cell]
            c0, c1 = c1, down_cell.out_channels
            down_channels.append(c1)
        down_channels.pop()
        for i in range(depth+1):
            c0 = down_channels.pop()
            up_cell = Cell(n_nodes, c0, c1, c_node, downward = False)
            self.up_cells += [up_cell]
            c1 = up_cell.out_channels
            c_node = c_node // 2 if channel_change else c_node  # halve the number of filters
        self.last_conv = nn.Sequential(ConvOps(c1, out_channels, kernel_size=1, 
                                               dropout_rate=0.1, ops_order='weight'),
                                       nn.Sigmoid())

    def forward(self, x, alpha1_down, alpha1_up, alpha2_down, alpha2_up):
        '''
        alpha1_down: Weights for downward MixedOps with stride == 1
        alpha1_up:   Weights for upward MixedOps with stride == 1
        alpha2_down: Weights for downward MixedOps with stride == 2
        alpha2_up:   Weights for upward MixedOps with stride == 2
        Note these alphas are different from the original alphas in ShellNet,
        they are the F.softmax(original alphas).
        '''
        s0, s1 = self.stem0(x), self.stem1(x)
        down_outputs = [s0, s1]
        for i, cell in enumerate(self.down_cells):
            s0, s1 = s1, cell(s0, s1, alpha1_down, alpha2_down)
            down_outputs.append(s1)
        if FLAG_DEBUG:
            print('x.shape = ',x.shape)
            for i in down_outputs: 
                print(i.shape)
        down_outputs.pop()
        for i, cell in enumerate(self.up_cells):
            s0 = down_outputs.pop()
            s1 = cell(s0, s1, alpha1_up, alpha2_up)
            if FLAG_DEBUG:
                print(s1.shape)
        return self.last_conv(s1)
    
    
class ShellNet(nn.Module):
    def __init__(self, in_channels, init_n_kernels, out_channels, depth, n_nodes,
                 normal_w_share=False, channel_change=False):
        '''
        This class defines the architectural params. I take it as the case/packing/box/shell of NAS. 
        in_channels: How many kinds of MRI modalities being used.
        init_n_kernels: Number of kernels for the nodes in the first cell.
        out_channels: How many kinds of tumor labels.
        depth: Number of downward cells. For upward, it has depth+1 cells.
        n_nodes: Number of nodes in each cell.
        normal_w_share: If True, self.alpha1_up = self.alpha1_down
        channel_change: If True, channel size expands and shrinks in double during downward and upward forwarding.  
        '''
        super().__init__()
        self.normal_w_share = normal_w_share
        self.n_nodes = n_nodes

        self.kernel = KernelNet(in_channels, init_n_kernels, out_channels, 
                             depth, n_nodes, channel_change)
        self._init_alphas()
        
    def _init_alphas(self):
        '''
        alpha2_down, alpha2_up: Weights for MixedOps with stride=2
        alpha1_down, alpha1_up: Weights for MixedOps with stride=1
        The 'down' and 'up' indicate the MixedOp is in the downward or upward blocks(cells).
        '''
        n_ops = sum(range(2, 2 + self.n_nodes))
        self.alpha2_down  = nn.Parameter(torch.zeros((n_ops, len(DownOps))))
        self.alpha2_up = nn.Parameter(torch.zeros((n_ops, len(UpOps))))
        self.alpha1_down = nn.Parameter(torch.zeros((n_ops, len(NormOps))))
        self.alpha1_up =  self.alpha1_down if self.normal_w_share else nn.Parameter(
                                    torch.zeros((n_ops, len(NormOps))))
        # setup alphas list
        self._alphas = [(name, param) for name, param in self.named_parameters() if 'alpha' in name]
        
    def alphas(self):
        for _, param in self._alphas:
            yield param

    def forward(self, x):
        return self.kernel(x, 
                           F.softmax(self.alpha1_down, dim=-1), 
                           F.softmax(self.alpha1_up, dim=-1), 
                           F.softmax(self.alpha2_down, dim=-1), 
                           F.softmax(self.alpha2_up, dim=-1))
    
    def get_gene(self):
        geno_parser = GenoParser(self.n_nodes)
        gene_down = geno_parser.parse(F.softmax(self.alpha1_down, dim=-1).detach().cpu().numpy(),
                           F.softmax(self.alpha2_down, dim=-1).detach().cpu().numpy())
        gene_up = geno_parser.parse(F.softmax(self.alpha1_up, dim=-1).detach().cpu().numpy(),
                         F.softmax(self.alpha2_up, dim=-1).detach().cpu().numpy(), downward=False)

        concat = range(2, self.n_nodes+2)
        return Genotype(down=gene_down, 
                        down_concat=concat,
                        up=gene_up, 
                        up_concat=concat)
    

        