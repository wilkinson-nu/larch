from torch import nn
import torch
import MinkowskiEngine as ME
from larch.models.resnetv2_blocks import PreActBasicBlock, PreActBottleneck
from larch.models.resnetv1_blocks import BasicBlock, Bottleneck
from collections import OrderedDict

## This is taken from the MinkowskiEngine implementation, but modified for v2 blocks
## Some inspiration + error checking against https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 256)

    ## In channels = 1 and outchannels are set by the network
    def __init__(self,
                 enc_act="relu",
                 stem_pool='none',
                 init_stem_stride=2,
                 final_stem_stride=2,
                 stem_norm=False,
                 stem_deep=False,
                 res_pool=False,
                 layer1_norm=True,
                 pool="avg",
                 bottleneck_dim=-1,
                 stem_channels=-1,
                 bn_momentum=0.1,
                 D=2):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.enc_act = enc_act
        self.stem_pool = stem_pool
        self.init_stem_stride = init_stem_stride
        self.final_stem_stride = final_stem_stride
        self.res_pool = res_pool
        self.stem_norm = stem_norm
        self.stem_deep = stem_deep
        self.layer1_norm = layer1_norm
        self.pool = pool
        self.bottleneck_dim = bottleneck_dim
        self.stem_channels = stem_channels
        self.bn_momentum = bn_momentum
        
        ## Pooling options
        if self.pool == "max":
            self.global_pool = ME.MinkowskiGlobalMaxPooling()
        elif self.pool == "avg":
            self.global_pool = ME.MinkowskiGlobalAvgPooling()
        elif self.pool == "both":
            self.global_pool_avg = ME.MinkowskiGlobalAvgPooling()
            self.global_pool_max = ME.MinkowskiGlobalMaxPooling()
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")

        if self.enc_act == "relu":
            self.act_fn = ME.MinkowskiReLU(inplace=True)
        if self.enc_act == "leakyrelu":
            self.act_fn =  ME.MinkowskiLeakyReLU()
        if self.enc_act == "gelu":
            self.act_fn = ME.MinkowskiGELU()
        if self.enc_act in ["silu", "swish"]:
            self.act_fn = ME.MinkowskiSiLU()

        ## Initialise the network
        self.network_initialization(D)
            
        ## Add the option for a bottleneck layer
        if self.bottleneck_dim > 0:
            self.bottleneck = nn.Linear(self.get_pool_nchan(), bottleneck_dim, bias=True)
        else:
            self.bottleneck = None

        ## Initialise all weights after everthing is built
        self.weight_initialization()
                    
            
    def make_shallow_stem(self):

        ## Bit of a hack
        if self.stem_channels < 0: self.stem_channels = self.INIT_DIM
            
        stem = OrderedDict()
        stem['conv1'] = ME.MinkowskiConvolution(in_channels=1, out_channels=self.stem_channels, kernel_size=3, stride=self.init_stem_stride, dimension=self.D)
        if self.stem_norm:
            stem['norm1'] = ME.MinkowskiBatchNorm(self.stem_channels, momentum=self.bn_momentum)
        stem['act1'] = self.act_fn

        ## Option of having a pooling layer or an extra downsampling convolution
        if self.stem_pool == 'max':
            stem['pool'] = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=self.D)
            self.stem_out_channels = self.stem_channels
        elif self.stem_pool == 'avg':
            stem['pool'] = ME.MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=self.D)
            self.stem_out_channels = self.stem_channels
        elif self.stem_pool == "none":
            stem['conv2'] = ME.MinkowskiConvolution(in_channels=self.stem_channels, out_channels=self.INIT_DIM, kernel_size=3, stride=self.final_stem_stride, dimension=self.D)

            self.stem_out_channels = self.INIT_DIM
            
            ## Allow for v1 style
            if self.BLOCK in (Bottleneck, BasicBlock):
                stem['act2'] = self.act_fn

        else:
            raise ValueError(f"Unknown stem_pool: {self.stem_pool}")
        
        return nn.Sequential(stem)
        
    def make_deep_stem(self):
        
        stem = OrderedDict()

        ## Bit of a hack
        if self.stem_channels < 0: self.stem_channels = self.INIT_DIM
        
        ## As is common for ResNet implementations, use 3 3x3 convoutions instead of an initial 7x7 one
        stem['conv1'] = ME.MinkowskiConvolution(in_channels=1, out_channels=self.stem_channels, kernel_size=3, stride=self.init_stem_stride, dimension=self.D)
        if self.stem_norm: stem['norm1'] = ME.MinkowskiBatchNorm(self.stem_channels, momentum=self.bn_momentum)
        stem['act1'] = self.act_fn
        stem['conv2'] = ME.MinkowskiConvolution(in_channels=self.stem_channels, out_channels=self.stem_channels, kernel_size=3, stride=1, dimension=self.D)
        if self.stem_norm: stem['norm2'] = ME.MinkowskiBatchNorm(self.stem_channels, momentum=self.bn_momentum)
        stem['act2'] = self.act_fn
        stem['conv3'] = ME.MinkowskiConvolution(in_channels=self.stem_channels, out_channels=self.stem_channels, kernel_size=3, stride=1, dimension=self.D)
        if self.stem_norm: stem['norm3'] = ME.MinkowskiBatchNorm(self.stem_channels, momentum=self.bn_momentum)
        stem['act3'] = self.act_fn
        
        ## Option of having a pooling layer or an extra downsampling convolution
        if self.stem_pool == "max":
            stem['pool'] = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=self.D)
            self.stem_out_channels = self.stem_channels
        elif self.stem_pool == "avg":
            stem['pool'] = ME.MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=self.D)
            self.stem_out_channels = self.stem_channels
        elif self.stem_pool == "none":
            stem['conv4'] = ME.MinkowskiConvolution(in_channels=self.stem_channels, out_channels=self.INIT_DIM, kernel_size=3, stride=self.final_stem_stride, dimension=self.D)
            self.stem_out_channels = self.INIT_DIM
            ## Allow for v1 blocks later
            if self.BLOCK in (Bottleneck, BasicBlock):
                stem['act4'] =	self.act_fn
        else:
            raise ValueError(f"Unknown stem_pool: {self.stem_pool}")
        
        return nn.Sequential(stem)

    
    def network_initialization(self, D):

        if self.stem_deep:
            self.stem = self.make_deep_stem()
        else:
            self.stem = self.make_shallow_stem()

        ## Input to the ResNet blocks set from the number of channels the stem produces
        self.inplanes = self.stem_out_channels
            
        ## In the original ME implementation, layer 1 has stride 2, which is nonstandard
        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=1
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        ## Note that I removed the conv5 from the original ME implementation, there was no need for such an aggressive downsampling for this purpose
        ## So removed for simplicity        

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # For v1, go back and zero out BN weights for the residual bit
        for m in self.modules():
            if isinstance(m, BasicBlock) and m.norm2 is not None:
                nn.init.zeros_(m.norm2.bn.weight)
            
            elif isinstance(m, Bottleneck) and m.norm3 is not None:
                nn.init.zeros_(m.norm3.bn.weight)


    def _make_layer(self, block, planes, num_blocks, stride, dilation=1):

        ## A bit of a hack to optionally remove BN from the first layer
        apply_bn = True
        if stride == 1: apply_bn = False
        if self.layer1_norm == True: apply_bn = True
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                res_pool=self.res_pool,
                apply_norm=apply_bn,
                enc_act=self.enc_act,
                bn_momentum = self.bn_momentum,
                dimension=self.D
            ))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, batch_size=None):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        ## Deal with optional concatenation of pooling options
        if self.pool == "both":
            x = torch.cat([self.global_pool_avg(x).F, self.global_pool_max(x).F], dim=-1)
        else:
            x = self.global_pool(x).F
            
        ## Add the bottleneck if requested
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        return x

    def get_pool_nchan(self):
        base = self.BLOCK.expansion * self.PLANES[-1]
        if self.pool == "both":
            base = base * 2
        return base

    def get_nchan(self):
        if self.bottleneck_dim > 0:
            return self.bottleneck_dim
        return self.get_pool_nchan()
    

class ResNet18v2(ResNetBase):
    BLOCK = PreActBasicBlock
    INIT_DIM=64
    LAYERS = (2, 2, 2, 2)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)

class ResNet34v2(ResNetBase):
    BLOCK = PreActBasicBlock
    INIT_DIM=64
    LAYERS = (3, 4, 6, 3)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)

class ResNet50v2(ResNetBase):
    BLOCK = PreActBottleneck
    INIT_DIM=64
    LAYERS = (3, 4, 6, 3)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)

## Sized for a 40GB GPU node
class ResNet101v2(ResNetBase):
    BLOCK = PreActBottleneck
    INIT_DIM=40
    LAYERS = (3, 4, 23, 3)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)

## Needs 80GB GPU node
class ResNet152v2(ResNetBase):
    BLOCK = PreActBottleneck
    INIT_DIM=48
    LAYERS = (3, 8, 36, 3)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)
    
class ResNet18v1(ResNetBase):
    BLOCK = BasicBlock
    INIT_DIM=64
    LAYERS = (2, 2, 2, 2)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)

class ResNet34v1(ResNetBase):
    BLOCK = BasicBlock
    INIT_DIM=64
    LAYERS = (3, 4, 6, 3)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)

class ResNet50v1(ResNetBase):
    BLOCK = Bottleneck
    INIT_DIM=64
    LAYERS = (3, 4, 6, 3)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)

## Sized for a 40GB GPU node
class ResNet101v1(ResNetBase):
    BLOCK = Bottleneck
    INIT_DIM=40
    LAYERS = (3, 4, 23, 3)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)

## Needs 80GB GPU node
class ResNet152v1(ResNetBase):
    BLOCK = Bottleneck
    INIT_DIM=48
    LAYERS = (3, 8, 36, 3)
    PLANES=(INIT_DIM, INIT_DIM*2, INIT_DIM*4, INIT_DIM*8)
    

def get_encoder(args):

    stem_norm = bool(getattr(args, "enc_stem_norm", 0))
    init_stem_stride = getattr(args, "enc_init_stem_stride", 2)
    final_stem_stride = getattr(args, "enc_final_stem_stride", 2)
    stem_pool = getattr(args, "enc_stem_pool", 'none')
    stem_deep = bool(getattr(args, "enc_stem_deep", 0))
    res_pool = bool(getattr(args, "enc_res_pool", 0))
    final_pool = getattr(args, "enc_arch_pool", "avg")
    layer1_norm = bool(getattr(args, "enc_layer1_norm", 1))
    bottleneck_dim = getattr(args, "enc_final_linear", -1)
    enc_act = getattr(args, "enc_act", "relu")
    stem_channels = getattr(args, "enc_stem_channels", -1)
    
    ## Only support ResNet for now 
    if "ResNet" not in args.enc_arch:
        raise ValueError(f"Unknown encoder architecture: {self.enc_arch}")
    
    if args.enc_arch in ["ResNet18", "ResNet18v2"]:
        enc = ResNet18v2
    elif args.enc_arch in ["ResNet34", "ResNet34v2"]:
        enc = ResNet34v2
    elif args.enc_arch in ["ResNet50", "ResNet50v2"]:
        enc = ResNet50v2
    elif args.enc_arch in ["ResNet101", "ResNet101v2"]:
        enc = ResNet101v2
    elif args.enc_arch in ["ResNet152", "ResNet152v2"]:
        enc = ResNet152v2
    elif args.enc_arch == "ResNet18v1":
        enc = ResNet18v1
    elif args.enc_arch == "ResNet34v1":
        enc = ResNet34v1
    elif args.enc_arch == "ResNet50v1":
        enc = ResNet50v1
    elif args.enc_arch == "ResNet101v1":
        enc = ResNet101v1
    elif args.enc_arch == "ResNet152v1":
        enc = ResNet152v1
        
    encoder = enc(enc_act=enc_act,
                  stem_pool=stem_pool,
                  init_stem_stride=init_stem_stride,
                  final_stem_stride=final_stem_stride,
	          stem_norm=stem_norm,
                  stem_deep=stem_deep,
                  res_pool=res_pool,
                  pool=final_pool,
                  layer1_norm=layer1_norm,
                  bottleneck_dim=bottleneck_dim,
                  stem_channels=stem_channels)
    
    return encoder

## Every args attribute get_encoder reads, plus the input resolution.
## A frozen encoder must be rebuilt from exactly these values.
ENCODER_ARG_KEYS = (
    "enc_act",
    "enc_arch",
    "enc_arch_pool",
    "enc_res_pool",
    "enc_stem_norm",
    "enc_init_stem_stride",
    "enc_final_stem_stride",
    "enc_stem_pool",
    "enc_stem_deep",
    "enc_layer1_norm",
    "enc_final_linear",
    "enc_stem_channels",
    "out_image_size",
)
