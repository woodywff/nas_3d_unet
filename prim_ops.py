import torch.nn as nn
# from util.utils import *
import pdb

OPS = {
'identity':      lambda c, stride: IdentityOp(c, c),
'se_conv':       lambda c, stride: SEConvOp(c, c),
'dil_conv':      lambda c, stride: ConvOps(c, c, dilation=2),
'dep_conv':      lambda c, stride: ConvOps(c, c, use_depthwise=True),
'conv':          lambda c, stride: ConvOps(c, c),
'avg_pool':      lambda c, stride: PoolingOp(c, c, pool_type='avg'),
'max_pool':      lambda c, stride: PoolingOp(c, c, pool_type='max'),
'down_se_conv':  lambda c, stride: SEConvOp(c, c, stride=2),
'down_dil_conv': lambda c, stride: ConvOps(c, c, stride=2, dilation=2),
'down_dep_conv': lambda c, stride: ConvOps(c, c, stride=2, use_depthwise=True),
'down_conv':     lambda c, stride: ConvOps(c, c, stride=2),
'up_se_conv':    lambda c, stride: SEConvOp(c, c, stride=2, use_transpose=True),
'up_dep_conv':   lambda c, stride: ConvOps(c, c, stride=2, use_depthwise=True, use_transpose=True),
'up_conv':       lambda c, stride: ConvOps(c, c, stride=2, use_transpose=True),
'up_dil_conv':   lambda c, stride: ConvOps(c, c, stride=2, dilation=2, use_transpose=True)
}
        
DownOps = [
'avg_pool',
'max_pool',
'down_se_conv',
'down_dil_conv',
'down_dep_conv',
'down_conv'
]

UpOps = [
'up_se_conv',
'up_dep_conv',
'up_conv',
'up_dil_conv'
]

NormOps = [
'identity',
'se_conv',
'dil_conv',
'dep_conv',
'conv'
]


class BaseOp(nn.Module):

    def __init__(self, in_channels, out_channels, 
                 dropout_rate=0, ops_order='weight_norm_act' ):
        super().__init__()
        self.ops_list = ops_order.split('_')
        
        # We use nn.GroupNorm cause our batch_size is too small.
        # Ref: https://arxiv.org/abs/1803.08494
        # 16 channels for one group is best
        if 'norm' in self.ops_list:
            group = 1 if out_channels % 16 != 0 else out_channels // 16
            self.norm = nn.GroupNorm(group, out_channels)
        else:
            self.norm = None

        # activation
        self.activation = nn.ReLU() if 'act' in self.ops_list else None

        # dropout
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == 'norm':
                if self.norm is not None:
                    x = self.norm(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise Warning('Unrecognized op: %s' % op)
        return x


class ConvOps(BaseOp):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, dilation=1, use_transpose=False, use_depthwise=False,
                 dropout_rate=0, ops_order='weight_norm_act'):
        
        super().__init__(in_channels, out_channels, dropout_rate, ops_order)
        
        self.use_depthwise = use_depthwise
        padding = (dilation * (kernel_size - 1) - stride + 1) // 2

        if use_transpose:
            if use_depthwise: # Ref: https://arxiv.org/abs/1704.04861
                self.depth_conv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size,
                                                     stride=stride, padding=padding, groups=in_channels)
                self.point_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            else: 
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
        else:
            if use_depthwise: 
                self.depth_conv = nn.Conv3d(in_channels, in_channels, kernel_size,
                                            stride=stride, padding=padding, groups=in_channels)
                self.point_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            else: 
                self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation)

    def weight_call(self, x):
        if self.use_depthwise:
            x = self.depth_conv(x)
            x = self.point_conv(x)
        else:
            x = self.conv(x)
        return x

class SEConvOp(BaseOp):
    '''
    Squeeze and Excitation Conv
    Ref: https://arxiv.org/abs/1709.01507
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, dilation=1, use_transpose=False,
                 dropout_rate=0, ops_order='weight_norm'):
        super().__init__(in_channels, out_channels, dropout_rate, ops_order)
        
        self.stride = stride
        padding = (dilation * (kernel_size - 1) - stride + 1) // 2

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.ReLU(),
            nn.Linear(1, out_channels),
            nn.Sigmoid()
        )
        
        if stride >= 2:
            if use_transpose:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding)
        else:
            self.norm = None

    def weight_call(self, x):
        batch_size, channels = x.size()[:2]
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1, 1)
        res = self.conv(x*y) if self.stride >= 2 else x*y
        return res


class PoolingOp(BaseOp):

    def __init__(self, in_channels, out_channels, pool_type, kernel_size=2, stride=2,
                 norm_type='gn', use_norm=False, affine=True, act_func=None, dropout_rate=0, ops_order='weight'):
        super(PoolingOp, self).__init__(in_channels, out_channels, norm_type, use_norm, affine, act_func, dropout_rate, ops_order)

        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        if self.stride == 1:
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        if self.pool_type == 'avg':
            self.pool = nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False)
        elif self.pool_type == 'max':
            self.pool = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError

    def weight_call(self, x):
        return self.pool(x)

class IdentityOp(BaseOp):

    def __init__(self, in_channels, out_channels, norm_type='gn', use_norm=False, affine=True,
                 act_func=None, dropout_rate=0, ops_order='weight_norm_act'):
        super(IdentityOp, self).__init__(in_channels, out_channels, norm_type,use_norm, affine,
                                          act_func, dropout_rate, ops_order)
    def weight_call(self, x):
        return x









