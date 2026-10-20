import torch
import torch.nn as nn
import MinkowskiEngine as ME


class MinkowskiGlobalGeM(nn.Module):
    """Generalized-mean global pooling: (mean_sites x^p)^(1/p) per event and channel.

    p=1 is mean pooling; p -> inf approaches max pooling.
    Expects non-negative inputs (e.g. post-ReLU); values are clamped to eps.
    """
    def __init__(self, p=3.0, eps=1e-6, learnable=False):
        super().__init__()
        p = torch.tensor(float(p))

        ## If the value of p is meant to be learned, register it
        if learnable:
            self.p = nn.Parameter(p)
        else:
            self.register_buffer("p", p)

        self.eps = eps
        self.avg = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        p = self.p.clamp(min=1.0)
        f = x.F.clamp(min=self.eps).pow(p)
        y = ME.SparseTensor(
            features=f,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        y = self.avg(y)
        return ME.SparseTensor(
            features=y.F.pow(1.0 / p),
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
        )

    def extra_repr(self):
        return f"p={float(self.p):.3f}, eps={self.eps}, learnable={isinstance(self.p, nn.Parameter)}"
