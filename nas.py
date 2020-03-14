import torch.nn as nn
from prim_ops import ConvOps, DownOps, UpOps, NormOps
from cell import Cell
from torch.functional import F

class KernelNet(nn.Module):
    def __init__(self, in_channels, init_n_kernels, out_channels, depth, n_nodes, channel_change):
        super().__init__()
        c0 = c1 = n_nodes * init_n_kernels # channel0, channel1, the number of kernels.
        c_node = init_n_kernels # channel dim doesn't change for different nodes

        self.stem0 = ConvOps(in_channels, c0, kernel_size=1, ops_order='weight_norm')
        self.stem1 = ConvOps(in_channels, c1, kernel_size=3,  stride=2, ops_order='weight_norm')

        assert depth >= 2 , 'depth must >= 2'

        self.down_cells = nn.ModuleList()
        self.up_cells = nn.ModuleList()

        down_channels = [c0, c1]
        for i in range(depth):
            c_node = 2 * c_node if channel_change else c_node  # double the number of filters
            down_cell = Cell(n_nodes, c0, c1, c_node, cell_type='down')
            self.down_cells += [down_cell]
            c0, c1 = c1, down_cell.out_channels
            down_channels.append(c1)
        down_channels.pop()
        for i in range(depth+1):
            c0 = down_channels.pop()
            up_cell = Cell(n_nodes, c0, c1, c_node, cell_type='up')
            self.up_cells += [up_cell]
            c1 = up_cell.out_channels
            c_node = c_node // 2 if channel_change else c_node  # halve the number of filters

        self.last_conv = ConvOps(c1, out_channels, kernel_size=1, 
                                 dropout_rate=0.1, ops_order='weight')

    def forward(self, x, w1_down, w1_up, w2_down, w2_up):
        '''
        w1_down: weights for downward MixedOp with stride == 1
        w1_up:   weights for upward MixedOp with stride == 1
        w2_down: weights for downward MixedOp with stride == 2
        w2_up:   weights for upward MixedOp with stride == 2
        '''
        s0, s1 = self.stem0(x), self.stem1(x)
        down_outputs = [s0, s1]
        for i, cell in enumerate(self.down_cells):
            s0, s1 = s1, cell(s0, s1, w1_down, w2_down)
            down_outputs.append(s1)
        down_outputs.pop()
        for i, cell in enumerate(self.up_cells):
            s0 = down_outputs.pop()
            s1 = cell(s0, s1, w1_up, w2_up)
        return self.last_conv(s1)
    
    
class NasShell(nn.Module):
    def __init__(self, in_channels, init_n_kernels, out_channels, depth, n_nodes,
                 device, normal_w_share=False, channel_change=False):
        '''
        This class defines the architectural params. I take it as the case/packing/box/shell of NAS. 
        in_channels: how many kinds of MRI modalities being used.
        init_n_kernels: number of kernels for the nodes in the first cell.
        out_channels: how many kinds of tumor labels.
        depth: number of downward cells. For upward, is depth+1
        n_nodes: number of nodes in each cell.
        normal_w_share: if True, self.alphas_normal_up = self.alphas_normal_down
        channel_change: if True, channel size expands and shrinks in double during downward and upward forwarding.  
        '''
        super().__init__()
        self.normal_w_share = normal_w_share
        self.n_nodes = n_nodes
#         self.device = device

        self.net = KernelNet(in_channels, init_n_kernels, out_channels, 
                             depth, n_nodes, channel_change)

        # Initialize architecture parameters: alpha
        self._init_alphas()
        
    def _init_alphas(self):
        '''
        alphas_down, alphas_up: params for MixedOp with stride=2
        alphas_normal_down, alphas_normal_up: params for MixedOp with stride=1
        '''
        n_ops = sum(range(2, 2 + self.n_nodes))
        self.alphas_down  = nn.Parameter(torch.zeros((n_ops, len(DownOps))))
        self.alphas_up = nn.Parameter(torch.zeros((n_ops, len(UpOps))))
        self.alphas_normal_down = nn.Parameter(torch.zeros((n_ops, len(NormOps))))
        self.alphas_normal_up =  self.alphas_normal_down if self.normal_w_share else nn.Parameter(
                                    torch.zeros((n_ops, len(NormOps))))


        # setup alphas list
        self._alphas = []
        for name, param in self.named_parameters():
            if 'alphas' in name: 
                self._alphas.append((name, param))
        
#         self._arch_parameters = [
#             self.alphas_normal_down,
#             self.alphas_down,
#             self.alphas_normal_up,
#             self.alphas_up
#         ]
        
    def alphas(self):
        for _, param in self._alphas:
            yield param

    def forward(self, x):
        w1_down = F.softmax(self.alphas_normal_down, dim=-1)
        w1_up = F.softmax(self.alphas_normal_up, dim=-1)
        w2_down = F.softmax(self.alphas_down, dim=-1)
        w2_up = F.softmax(self.alphas_up, dim=-1)
        return self.net(x, w1_down, w1_up, w2_down, w2_up)