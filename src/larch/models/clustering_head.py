from torch import nn
from larch.models.activations import get_act_from_string
from collections import OrderedDict


class ClusteringHeadLogits(nn.Module):
    def __init__(self, 
                 nchan : int, 
                 nclusters : int,
                 nhidden : int = -1,
                 hidden_act_fn : object = nn.ReLU,
                 apply_bn : bool = False):
        super().__init__()

        self.nchan = nchan
        self.nclusters = nclusters
        self.nhidden = nhidden if nhidden != -1 else nchan//4
        self.apply_bn = apply_bn
        self.hidden_act_fn = hidden_act_fn
        
        self.clust = self.make_network()
        self.initialize_weights()

    def make_network(self):
        clust = OrderedDict()
        clust['lin1'] = nn.Linear(self.nchan, self.nhidden, bias=True)
        if self.apply_bn: clust['norm1'] = nn.BatchNorm1d(self.nhidden)
        clust['act1'] = self.hidden_act_fn()
        clust['lin2'] = nn.Linear(self.nhidden, self.nclusters, bias=False)
        return nn.Sequential(clust)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, return_hidden=False):

        ## Separate execution so hidden layers can be returned
        h1_logits = self.clust.lin1(x)
        h1 = h1_logits

        if self.apply_bn: h1 = self.clust.norm1(h1)
        h1 = self.clust.act1(h1)
        h2 = self.clust.lin2(h1)

        if return_hidden:
            return {"clust_layer1": h1_logits,
                    "clust_final": h2}
        return h2

    
class ClusteringHeadTwoLayer(nn.Module):
    def __init__(self, 
                 nchan : int, 
                 nclusters : int,
                 nhidden : int = -1,
                 softmax_temp : float = 1.0,
                 hidden_act_fn : object = nn.ReLU,
                 apply_bn : bool = False):
        super().__init__()

        self.nchan = nchan
        self.nclusters = nclusters
        self.nhidden = nhidden if nhidden != -1 else nchan//4
        self.softmax_temp = softmax_temp
        self.apply_bn = apply_bn
        self.hidden_act_fn = hidden_act_fn
        
        self.clust = self.make_network()
        self.initialize_weights()
        self.softmax = nn.Softmax(dim=1)

    def make_network(self):
        clust = OrderedDict()
        clust['lin1'] = nn.Linear(self.nchan, self.nhidden, bias=True)
        if self.apply_bn: clust['norm1'] = nn.BatchNorm1d(self.nhidden)
        clust['act1'] = self.hidden_act_fn()
        clust['lin2'] = nn.Linear(self.nhidden, self.nclusters, bias=False)
        return nn.Sequential(clust)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, return_hidden=False):

        ## Separate execution so hidden layers can be returned
        h1_logits = self.clust.lin1(x)
        h1 = h1_logits

        if self.apply_bn: h1 = self.clust.norm1(h1)
        h1 = self.clust.act1(h1)
        h2 = self.clust.lin2(h1)
        out = self.softmax(h2/self.softmax_temp)

        if return_hidden:
            return {"clust_layer1": h1_logits,
                    "clust_final": out}
        return out

    
class ClusteringHeadThreeLayer(nn.Module):
    def __init__(self,
                 nchan : int,
                 nclusters : int,
                 nhidden : int = -1,
                 softmax_temp : float = 1.0,
                 hidden_act_fn : object = nn.ReLU,
	         apply_bn : bool = False):
        super().__init__()

        self.nchan = nchan
        self.nclusters = nclusters
        self.nhidden = nhidden if nhidden != -1 else nchan//4
        self.softmax_temp = softmax_temp
        self.apply_bn = apply_bn
        self.hidden_act_fn = hidden_act_fn
        
        self.clust = self.make_network()
        self.initialize_weights()
        self.softmax = nn.Softmax(dim=1)

    def make_network(self):
        clust = OrderedDict()
        clust['lin1'] = nn.Linear(self.nchan, self.nhidden, bias=True)
        if self.apply_bn: clust['norm1'] = nn.BatchNorm1d(self.nhidden)
        clust['act1'] = self.hidden_act_fn()
        clust['lin2'] = nn.Linear(self.nhidden, self.nhidden, bias=True)
        if self.apply_bn: clust['norm2'] = nn.BatchNorm1d(self.nhidden)
        clust['act2'] = self.hidden_act_fn()
        clust['lin3'] = nn.Linear(self.nhidden, self.nclusters, bias=False)
        return nn.Sequential(clust)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_hidden=False):

	## Separate execution so hidden layers can be returned
        h1_logits = self.clust.lin1(x)
        h1 = h1_logits

        if self.apply_bn: h1 = self.clust.norm1(h1)
        h1 = self.clust.act1(h1)
        h2_logits = self.clust.lin2(h1)
        h2 = h2_logits

        if self.apply_bn: h2 = self.clust.norm1(h2)
        h2 = self.clust.act2(h2)
        h3_logits = self.clust.lin3(h2)
        out = self.softmax(h3_logits/self.softmax_temp)

        if return_hidden:
            return {"clust_layer1": h1_logits,
                    "clust_layer2": h2_logits,
                    "clust_final": out}
        return out

    
    
class ClusteringHeadOneLayer(nn.Module):
    def __init__(self,
                 nchan : int,
                 nclusters : int,
                 softmax_temp : float):
        super().__init__()

        self.softmax_temp = softmax_temp
        self.linear = nn.Linear(nchan, nclusters)
        self.softmax = nn.Softmax(dim=1)        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="linear")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_hidden=False):
        x = self.linear(x)
        x = self.softmax(x/self.softmax_temp)

        if return_hidden:
            return {"clust_final": x}
        
        return x

def get_clusthead(nchan, args):

    hidden_act_fn = get_act_from_string(args.enc_act)
    if args.clust_arch == "none":
        clust_head = None
    elif args.clust_arch == "one":
        clust_head = ClusteringHeadOneLayer(nchan, args.nclusters, args.clust_temp)
    elif args.clust_arch == "twobn":
        clust_head = ClusteringHeadTwoLayer(nchan, args.nclusters, getattr(args, "nhidden", -1), args.clust_temp, hidden_act_fn, apply_bn=True)
    elif args.clust_arch == "two":
        clust_head = ClusteringHeadTwoLayer(nchan, args.nclusters, getattr(args, "nhidden", -1), args.clust_temp, hidden_act_fn, apply_bn=False)
    elif args.clust_arch == "threebn":
        clust_head = ClusteringHeadThreeLayer(nchan, args.nclusters, getattr(args, "nhidden", -1), args.clust_temp, hidden_act_fn, apply_bn=True)
    elif args.clust_arch == "three":
        clust_head = ClusteringHeadThreeLayer(nchan, args.nclusters, getattr(args, "nhidden", -1), args.clust_temp, hidden_act_fn, apply_bn=False)
    elif args.clust_arch == "logits":
        clust_head = ClusteringHeadLogits(nchan, args.nclusters, getattr(args, "nhidden", -1), hidden_act_fn, apply_bn=False)
        
    return clust_head
