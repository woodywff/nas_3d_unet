import torch
import torch.nn as nn
from prim_ops import ConvOps, DownOps, UpOps, NormOps
from cell import Cell
from torch.functional import F
import pdb



FLAG_DEBUG = True

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
        if FLAG_DEBUG:
            for i in down_outputs: 
                print(i.shape)
        down_outputs.pop()
        for i, cell in enumerate(self.up_cells):
            s0 = down_outputs.pop()
            s1 = cell(s0, s1, w1_up, w2_up)
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
        normal_w_share: If True, self.alphas_normal_up = self.alphas_normal_down
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
        self._alphas = [(name, param) for name, param in self.named_parameters() if 'alpha' in name]
        
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
        return self.kernel(x, w1_down, w1_up, w2_down, w2_up)
    
    
class ShellConsole:
    def __init__(self, shell_net, optim_shell, loss):
        '''
        This is where we define the step() method for ShellNet().alphas() update.
        shell_net: ShellNet() instance 
        optim_shell: Optimizer for shell_net
        loss: Loss function of shell_net
        '''
        self.shell_net = shell_net
        self.optimizer = optim_shell
        self.loss = loss

    def step(self, x, y_truth):
        '''
        Do one step of gradient descent for shell_net alphas.
        x: Input batch; shape: (batch_size, n_modalities, patch_size[0], patch_size[1], patch_size[2])
        y_truth: Label batch; shape: ((batch_size, n_labels, patch_size[0], patch_size[1], patch_size[2]))
        '''
#         pdb.set_trace()
#         from copy import deepcopy
#         t0 = deepcopy(self.shell_net.alphas_dict())

        self.optimizer.zero_grad()
        y_pred = self.shell_net(x)
        loss = self.loss(y_pred, y_truth)
        loss.backward()
        self.optimizer.step()
        
#         pdb.set_trace()
#         t1 = deepcopy(self.shell_net.alphas_dict())
#         for key in t1.keys():
#             print(t1[key] - t0[key])