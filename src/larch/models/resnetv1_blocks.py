## These blocks are based on the resnet block definitions in MinkowskiEngine:
## https://github.com/NVIDIA/MinkowskiEngine/blob/master/MinkowskiEngine/modules/resnet_block.py
## I also took the ResNet D option from https://arxiv.org/pdf/1812.01187 for the bottleneck block
import torch.nn as nn
import MinkowskiEngine as ME
from collections import OrderedDict

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 res_pool=False,
                 apply_norm=True,
                 enc_act='relu',
                 dimension=2):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        ## Make BN optional
        self.apply_norm = apply_norm
        self.enc_act = enc_act
        
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        if self.apply_norm:
            self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        else:
            self.norm1 = None

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        if self.apply_norm:
            self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        else:
            self.norm2 = None
            
        if self.enc_act == "relu":
            self.act_fn = ME.MinkowskiReLU(inplace=False)
        elif self.enc_act == "leakyrelu":
            self.act_fn =  ME.MinkowskiLeakyReLU()
        elif self.enc_act == "gelu":
            self.act_fn = ME.MinkowskiGELU()
        elif self.enc_act in ["silu", "swish"]:
            self.act_fn = ME.MinkowskiSiLU()
        else:
            raise ValueError(f"Unknown activation: {enc_act}")
        
        ## Support a few options for the residual connection
        if stride != 1 or inplanes != self.expansion*planes:
            res = OrderedDict()
            if res_pool and stride != 1:
                res['res_pool'] = ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=dimension)
                res['res_conv'] = ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=1, dimension=dimension)
            else:
                res['res_conv'] = ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=stride, dimension=dimension)
            if self.apply_norm:
                res['res_norm'] = ME.MinkowskiBatchNorm(self.expansion*planes, momentum=bn_momentum)
                ## Avoid duplicating these parameters in the sum
                res['res_norm'].bn.bias.requires_grad_(False)
            self.shortcut = nn.Sequential(res)
        else:
            self.shortcut = None

    ## Info on residual norms as these get init gamma = 0
    @property
    def residual_norm(self):
        return self.norm2
        
    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(x)
        if self.norm1 is not None: out = self.norm1(out)
        out = self.act_fn(out)
        out = self.conv2(out)
        if self.norm2 is not None: out = self.norm2(out)
        out += residual
        out = self.act_fn(out)
        return out
                

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 res_pool=False,
                 apply_norm=True,
                 enc_act='relu',
                 dimension=2):
        super(Bottleneck, self).__init__()
        assert dimension > 0

        ## Make BN optional
        self.apply_norm	= apply_norm
        self.enc_act = enc_act
        
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension)
        if self.apply_norm:
            self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        else:
            self.norm1 = None

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)

        if self.apply_norm:
            self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        else:
            self.norm2 = None

        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension)

        if self.apply_norm:
            self.norm3 = ME.MinkowskiBatchNorm(planes * self.expansion, momentum=bn_momentum)
        else:
            self.norm3 = None

        if self.enc_act == "relu":
            self.act_fn = ME.MinkowskiReLU(inplace=False)
        elif self.enc_act == "leakyrelu":
            self.act_fn =  ME.MinkowskiLeakyReLU()
        elif self.enc_act == "gelu":
            self.act_fn = ME.MinkowskiGELU()
        elif self.enc_act in ["silu", "swish"]:
            self.act_fn = ME.MinkowskiSiLU()
        else:
            raise ValueError(f"Unknown activation: {enc_act}")

        ## Support a few options for the residual connection
        if stride != 1 or inplanes != self.expansion*planes:
            res = OrderedDict()
            if res_pool and stride != 1:
                res['res_pool'] = ME.MinkowskiAvgPooling(kernel_size=stride, stride=stride, dimension=dimension)
                res['res_conv'] = ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=1, dimension=dimension)
            else:
                res['res_conv'] = ME.MinkowskiConvolution(inplanes, self.expansion*planes, kernel_size=1, stride=stride, dimension=dimension)
            if self.apply_norm:
                res['res_norm'] = ME.MinkowskiBatchNorm(self.expansion*planes, momentum=bn_momentum)
                ## Avoid duplicating these parameters in the sum
                res['res_norm'].bn.bias.requires_grad_(False)
            self.shortcut = nn.Sequential(res)
        else:
            self.shortcut = None

    ## Info on residual norms as these get init gamma = 0
    @property
    def residual_norm(self):
        return self.norm3
    
    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(x)
        if self.norm1 is not None: out = self.norm1(out)
        out = self.act_fn(out)
        out = self.conv2(out)
        if self.norm2 is not None: out = self.norm2(out)
        out = self.act_fn(out)
        out = self.conv3(out)
        if self.norm3 is not None: out = self.norm3(out)
        out += residual
        out = self.act_fn(out)
        return out
