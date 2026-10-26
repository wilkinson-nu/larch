from torch import nn
from larch.models.activations import get_act_from_string
from collections import OrderedDict

class ProjectionHeadOneLogits(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int):
        super().__init__()

        self.proj = nn.Linear(nchan, nlatent, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)

    ## return_logits does nothing for a single layer
    def forward(self, x, return_hidden=False):
        if return_hidden:
            return {"proj_final": self.proj(x)}
        return self.proj(x)

class ProjectionHeadTwoLayer(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int,
                 hidden_act_fn : object = nn.ReLU,
                 apply_bn : bool = False,
                 affine_bn : bool = True,
                 init_bn : bool = False,
                 final_bn : bool = False):
        super().__init__()

        self.nchan = nchan
        self.nlatent = nlatent
        self.nhidden = nhidden
        self.apply_bn = apply_bn
        self.affine_bn = affine_bn
        self.init_bn = init_bn
        self.final_bn = final_bn
        self.hidden_act_fn = hidden_act_fn
        
        self.proj = self.make_network()
        self.initialize_weights()

    def make_network(self):
        proj = OrderedDict()
        if self.init_bn:
            proj['norm0'] = nn.BatchNorm1d(self.nchan, affine=self.affine_bn)
            proj['act0'] = self.hidden_act_fn()
            
        proj['lin1'] = nn.Linear(self.nchan, self.nhidden, bias=not self.apply_bn)
        if self.apply_bn:
            proj['norm1'] = nn.BatchNorm1d(self.nhidden, affine=self.affine_bn)
        proj['act1'] = self.hidden_act_fn()

        proj['lin2'] = nn.Linear(self.nhidden, self.nlatent, bias=False)
        if self.apply_bn and self.final_bn:
            proj['norm2'] = nn.BatchNorm1d(self.nlatent, affine=False)

        return nn.Sequential(proj)

    
    def initialize_weights(self):
        act = self.hidden_act_fn()
        slope = getattr(act, "negative_slope", 0.0)
        nl = "leaky_relu" if slope > 0 else "relu"

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nl, a=slope)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x, return_hidden=False):

        ## Separate execution so hidden layers can be returned
        if self.init_bn:
            x = self.proj.norm0(x)
            x = self.proj.act0(x)

        h1 = self.proj.lin1(x)
        if self.apply_bn: h1 = self.proj.norm1(h1)
        h1 = self.proj.act1(h1)

        out = self.proj.lin2(h1)
        if self.apply_bn and self.final_bn:
            out = self.proj.norm2(out)

        if return_hidden:
            return {"proj_layer1": h1,
                    "proj_final": out}
        return out


class ProjectionHeadThreeLayer(nn.Module):
    def __init__(self,
                 nchan : int,
                 nlatent: int,
                 nhidden: int,
                 hidden_act_fn : object = nn.ReLU,
                 apply_bn : bool = False,
                 affine_bn : bool = True,
                 init_bn : bool = False,
                 final_bn : bool = False):
        super().__init__()


        self.nchan = nchan
        self.nlatent = nlatent
        self.nhidden = nhidden
        self.apply_bn = apply_bn
        self.affine_bn = affine_bn
        self.init_bn = init_bn
        self.final_bn = final_bn
        self.hidden_act_fn = hidden_act_fn

        self.proj = self.make_network()
        self.initialize_weights()

    def make_network(self):
        proj = OrderedDict()
        if self.init_bn:
            proj['norm0'] = nn.BatchNorm1d(self.nchan, affine=self.affine_bn)
            proj['act0'] = self.hidden_act_fn()
                
        proj['lin1'] = nn.Linear(self.nchan, self.nhidden, bias=not self.apply_bn)
        if self.apply_bn:
            proj['norm1'] = nn.BatchNorm1d(self.nhidden, affine=self.affine_bn)
        proj['act1'] = self.hidden_act_fn()

        proj['lin2'] = nn.Linear(self.nhidden, self.nhidden, bias=not self.apply_bn)
        if self.apply_bn:
            proj['norm2'] = nn.BatchNorm1d(self.nhidden, affine=self.affine_bn)
        proj['act2'] = self.hidden_act_fn()

        proj['lin3'] = nn.Linear(self.nhidden, self.nlatent, bias=False)
        if self.apply_bn and self.final_bn:
            proj['norm3'] = nn.BatchNorm1d(self.nlatent, affine=False)

        return nn.Sequential(proj)
        

    def initialize_weights(self):

        act = self.hidden_act_fn()
        slope = getattr(act, "negative_slope", 0.0)
        nl = "leaky_relu" if slope > 0 else "relu"

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nl, a=slope)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x, return_hidden=False):

        ## Separate execution so hidden layers can be returned
        if self.init_bn:
            x = self.proj.norm0(x)
            x = self.proj.act0(x)

        h1 = self.proj.lin1(x)
        if self.apply_bn: h1 = self.proj.norm1(h1)
        h1 = self.proj.act1(h1)

        h2 = self.proj.lin2(h1)
        if self.apply_bn: h2 = self.proj.norm2(h2)
        h2 = self.proj.act2(h2)

        out = self.proj.lin3(h2)
        if self.apply_bn and self.final_bn:
            out = self.proj.norm3(out)

        if return_hidden:
            return {"proj_layer1": h1,
                    "proj_layer2": h2,
                    "proj_final": out}
        return out
        
        
def get_projhead(nchan, args):

    ## Temporary protection for old files that don't have split activation functions
    proj_act = getattr(args, "proj_act", args.enc_act)

    hidden_act_fn = get_act_from_string(proj_act)
    
    if args.proj_arch in ["logits", "two"]:
        proj_head = ProjectionHeadTwoLayer(nchan, args.latent, args.nhidden, hidden_act_fn,
                                           apply_bn=False, init_bn=args.proj_init_bn, final_bn=args.proj_final_bn)
    elif args.proj_arch in ["logitsbn", "twobn"]:
        proj_head = ProjectionHeadTwoLayer(nchan, args.latent, args.nhidden, hidden_act_fn,
                                           apply_bn=True, init_bn=args.proj_init_bn, final_bn=args.proj_final_bn)
    elif args.proj_arch == "twobnnoaffine":
        proj_head = ProjectionHeadTwoLayer(nchan, args.latent, args.nhidden, hidden_act_fn, affine_bn=False,
                                           apply_bn=True, init_bn=args.proj_init_bn, final_bn=args.proj_final_bn)
    elif args.proj_arch == "three":
        proj_head = ProjectionHeadThreeLayer(nchan, args.latent, args.nhidden, hidden_act_fn,
                                             apply_bn=False, init_bn=args.proj_init_bn, final_bn=args.proj_final_bn)
    elif args.proj_arch == "threebn":
        proj_head = ProjectionHeadThreeLayer(nchan, args.latent, args.nhidden, hidden_act_fn,
                                             apply_bn=True, init_bn=args.proj_init_bn, final_bn=args.proj_final_bn)
    elif args.proj_arch == "threebnnoaffine":
        proj_head = ProjectionHeadThreeLayer(nchan, args.latent, args.nhidden, hidden_act_fn, affine_bn=False,
                                             apply_bn=True, init_bn=args.proj_init_bn, final_bn=args.proj_final_bn)
    elif args.proj_arch == "one":
        proj_head = ProjectionHeadOneLayer(nchan, args.latent)
    return proj_head
