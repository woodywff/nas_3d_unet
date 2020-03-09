import torch.nn as nn
# from util.utils import *
import pdb

OPS = {
    'none': lambda c, stride, affine, dp: ZeroOp(c, c, stride=stride),
    'identity': lambda c, stride, affine, dp: IdentityOp(c, c, affine=affine),
    'cweight': lambda c, stride, affine, dp: CWeightOp(c, c, affine=affine, dropout_rate=dp),
    'dil_conv': lambda c, stride, affine, dp: ConvOps(c, c, affine=affine, dilation=2, dropout_rate=dp),
    'dep_conv': lambda c, stride, affine, dp: ConvOps(c, c, affine=affine, use_depthwise=True, dropout_rate=dp),
    'shuffle_conv': lambda c, stride, affine, dp: ConvOps(c, c, affine=affine),# ConvOps(c, c, affine=affine, has_shuffle=True)
    'conv': lambda c, stride, affine, dp: ConvOps(c, c, affine=affine, has_shuffle=True),
    'avg_pool': lambda c, stride, affine, dp: PoolingOp(c, c, affine=affine, pool_type='avg'),
    'max_pool': lambda c, stride, affine, dp: PoolingOp(c, c, affine=affine,pool_type='max'),
    'down_cweight': lambda c, stride, affine, dp: CWeightOp(c, c, stride=2, affine=affine, dropout_rate=dp),
    'down_dil_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, dilation=2, dropout_rate=dp),
    'down_dep_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, use_depthwise=True, dropout_rate=dp),
    'down_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, dropout_rate=dp),
    'up_cweight': lambda c, stride, affine, dp: CWeightOp(c, c, stride=2, affine=affine,use_transpose=True, dropout_rate=dp),
    'up_dep_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine,use_depthwise=True, use_transpose=True, dropout_rate=dp),
    'up_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, use_transpose=True, dropout_rate=dp),
    'up_dil_conv': lambda c, stride, affine, dp: ConvOps(c, c, stride=2, affine=affine, dilation=2,use_transpose=True,  dropout_rate=dp),
}

DownOps = [
    'avg_pool',
    'max_pool',
    'down_cweight',
    'down_dil_conv',
    'down_dep_conv',
    'down_conv'
]

UpOps = [
    'up_cweight',
    'up_dep_conv',
    'up_conv',
    'up_dil_conv'
]

NormOps = [
    'identity',
    'cweight',
    'dil_conv',
    'dep_conv',
    'conv',
]




class BaseOp(nn.Module):

    def __init__(self, in_channels, out_channels, norm_type='gn', use_norm=True, affine=True,
                 act_func='relu', dropout_rate=0, ops_order='weight_norm_act' ):
        super().__init__()

        self.ops_order = ops_order
        

        # batch norm, group norm, instance norm, layer norm
        if use_norm:
            # Ref: <Group Normalization> https://arxiv.org/abs/1803.08494
            # 16 channels for one group is best
            if self.norm_before_weight:
                group = 1 if in_channels % 16 != 0 else in_channels // 16
                if norm_type == 'gn':
                    self.norm = nn.GroupNorm(group, in_channels, affine=affine)
                else:
                    self.norm = nn.BatchNorm2d(in_channels, affine=affine)
            else:
                group = 1 if out_channels % 16 != 0 else out_channels // 16
                if norm_type == 'gn':
                    self.norm = nn.GroupNorm(group, out_channels, affine=affine)
                else:
                    self.norm = nn.BatchNorm2d(out_channels, affine=affine)
        else:
            self.norm = None

        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU(inplace=False)
            else:
                self.activation = nn.ReLU(inplace=True)
        elif act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU6(inplace=False)
            else:
                self.activation = nn.ReLU6(inplace=True)
        else:
            self.activation = None

        # dropout
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate, inplace=False)
        else:
            self.dropout = None

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def norm_before_weight(self):
        for op in self.ops_list:
            if op == 'norm':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

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
                raise ValueError('Unrecognized op: %s' % op)
        return x


class ConvOps(BaseOp):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,dilation=1,groups=1,
                 bias=False, use_transpose=False, output_padding=0, use_depthwise=False,
                 norm_type='gn', use_norm=True, affine=True, act_func='relu', dropout_rate=0,
                 ops_order='weight_norm_act'):
        
        super().__init__(in_channels, out_channels, norm_type, 
                         use_norm, affine, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_transpose = use_transpose
        self.use_depthwise = use_depthwise
        self.output_padding = output_padding
        
        return

        # I'm afraid it only works for dilation=1
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        # 'kernel_size', 'stride', 'padding', 'dilation' can either be 'int' or 'tuple' of int
        if use_transpose:
            if use_depthwise: # 1. transpose depth-wise conv
                self.depth_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=self.kernel_size,
                        stride=self.stride, padding=padding, output_padding=self.output_padding, groups=in_channels, bias=self.bias)
                self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                            groups=self.groups, bias=False)
            else: # 2. transpose conv
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                            stride=self.stride, padding=padding,
                            output_padding=self.output_padding, dilation=self.dilation, bias=self.bias)
        else:
            if use_depthwise: # 3. depth-wise conv
                self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size,
                        stride=self.stride, padding=padding,
                        dilation=self.dilation, groups=in_channels, bias=False)
                self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                            groups=self.groups, bias=False)
            else: # 4. conv
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size,
                            stride=self.stride, padding=padding,
                            dilation=self.dilation, bias=False)

    def weight_call(self, x):
        if self.use_depthwise:
            x = self.depth_conv(x)
            x = self.point_conv(x)
        else:
            x = self.conv(x)
        if self.has_shuffle and self.groups > 1:
            pdb.set_trace()
            x = shuffle_layer(x, self.groups)
        return x

class CWeightOp(BaseOp):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,dilation=1, groups=None,
                 bias=False, has_shuffle=False, use_transpose=False,output_padding=0, norm_type='gn',
                 use_norm=False, affine=True, act_func=None, dropout_rate=0, ops_order='weight'):
        super(CWeightOp, self).__init__(in_channels, out_channels, norm_type, use_norm, affine, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_transpose = use_transpose
        self.output_padding = output_padding

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        # `kernel_size`, `stride`, `padding`, `dilation`
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, out_channels),
            nn.Sigmoid()
        )
        if stride >= 2:
            if use_transpose:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                               stride=self.stride, padding=padding, output_padding=self.output_padding,
                                                bias=False)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=False)
            group = 1 if out_channels % 16 != 0 else out_channels // 16
            self.norm = nn.GroupNorm(group, out_channels, affine=affine)


    def weight_call(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        rst = self.norm(self.conv(x*y)) if self.stride >= 2 else x*y
        return rst


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









