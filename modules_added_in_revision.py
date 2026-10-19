"""Modules added in the revision (JOE 26-0110).  Append these classes to pwst_modules.py / ultralytics/nn/modules/block.py, import them in
ultralytics/nn/tasks.py and list the convolution types (ConvGIN0, ConvGIN5, ConvSRAL, ConvECA, ConvSE) in `base_modules` of parse_model.

The technical improvement of the revision is the gated input standardization (GatedIN, used as ConvGIN5 in place of the first Conv of YOLOv11n; configs/x_gin5.yaml).
All other classes are candidates of the search that is documented in the Supplementary Material (S2-S4)."""
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv, PWAM, SRA  # noqa: F401  (PWAM and SRA are defined in pwst_modules.py)


class GatedIN(nn.Module):
    """x + a * (IN(x) * s + m - x): per-image, per-channel standardisation (instance normalisation) re-scaled to the typical dataset statistics (m, s),
    blended with a learnable per-channel gate a (a0 = 0 would be the identity at initialisation; the technical improvement uses a0 = 0.5).
    Removes per-image colour cast and contrast differences (e.g. red-channel loss, haze)."""

    def __init__(self, c, a0=0.0, m=0.4, s=0.18, eps=1e-3):
        super().__init__()
        self.gate_bias = nn.Parameter(torch.full((c,), float(a0)))
        self.m, self.s, self.eps = m, s, eps

    def forward(self, x):
        xf = x.float()
        mu = xf.mean(dim=(2, 3), keepdim=True)
        sd = (xf.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt()
        y = (xf - mu) / sd * self.s + self.m
        a = self.gate_bias.float().view(1, -1, 1, 1)
        return (xf + a * (y - xf)).to(x.dtype)


class ConvGIN0(Conv):
    """Conv whose input passes through GatedIN (a0 = 0)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.gin = GatedIN(c1, 0.0)

    def forward(self, x):
        return super().forward(self.gin(x))

    def forward_fuse(self, x):
        return super().forward_fuse(self.gin(x))


class ConvGIN5(ConvGIN0):
    """Conv whose input passes through GatedIN (a0 = 0.5): the technical improvement of the revision (x_gin5)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.gin = GatedIN(c1, 0.5)


class ConvSRAL(Conv):
    """ConvSRAW plus a zero-gated depth-wise 7x7 branch: y + g1 * SRA(y) + g2 * DW7(y), g1 = g2 = 0 at initialisation."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True, heads=4, red=2):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.pw = PWAM(c2, c2)
        self.sra = SRA(c2, num_heads=heads, reduction_ratio=red)
        self.dw = nn.Sequential(nn.Conv2d(c2, c2, 7, 1, 3, groups=c2, bias=False), nn.BatchNorm2d(c2))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_dw = nn.Parameter(torch.zeros(1))

    def _tail(self, y):
        y = self.pw(y)
        return y + self.gamma * self.sra(y) + self.gamma_dw * self.dw(y)

    def forward(self, x):
        return self._tail(super().forward(x))

    def forward_fuse(self, x):
        return self._tail(super().forward_fuse(x))


class ConvECA(Conv):
    """Conv followed by zero-gated efficient channel attention (4-D weight: the Muon optimizer of the fork needs 2-D or 4-D weights)."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True, ks=5):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.eca = nn.Conv2d(1, 1, (ks, 1), padding=(ks // 2, 0), bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _tail(self, y):
        w = y.mean(dim=(2, 3)).unsqueeze(1).unsqueeze(-1)
        w = torch.sigmoid(self.eca(w)).squeeze(-1).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return y + self.gamma * y * (2.0 * w - 1.0)

    def forward(self, x):
        return self._tail(super().forward(x))

    def forward_fuse(self, x):
        return self._tail(super().forward_fuse(x))


class ConvSE(Conv):
    """Conv followed by zero-gated squeeze-and-excitation (reduction 8)."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True, r=8):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.fc = nn.Sequential(nn.Linear(c2, max(c2 // r, 4)), nn.SiLU(), nn.Linear(max(c2 // r, 4), c2))
        self.gamma = nn.Parameter(torch.zeros(1))

    def _tail(self, y):
        w = torch.sigmoid(self.fc(y.mean(dim=(2, 3)))).unsqueeze(-1).unsqueeze(-1)
        return y + self.gamma * y * (2.0 * w - 1.0)

    def forward(self, x):
        return self._tail(super().forward(x))

    def forward_fuse(self, x):
        return self._tail(super().forward_fuse(x))
