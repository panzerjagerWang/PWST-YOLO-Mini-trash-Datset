"""Modules of the PWST-YOLO revision (extracted from the Ultralytics fork used for the experiments).
Place these classes in ultralytics/nn/modules/block.py, import them in ultralytics/nn/tasks.py and add the convolution-type
modules (ConvPWAM, ConvSRAW, ConvSRA, ConvAmp0, ConvWPATM, C3k2PW, C3k2PWSRA, C3k2MS) to `base_modules` (and the C3k2 subclasses to
`repeat_modules`) of parse_model.  PWAM_MODE=full|amp|phase (environment variable) selects the amplitude/phase ablation of PWAM."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, C3k2  # noqa: F401



class PWAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c2 = c2

        # 1. Input projection
        self.amp_conv = Conv(c1, c2, 1, 1)
        self.phase_conv = Conv(c1, c2, 1, 1)

        # 2. Complex Weights
        self.weight_real = nn.Parameter(torch.Tensor(c2, c2))
        self.weight_imag = nn.Parameter(torch.Tensor(c2, c2))

        # 3. Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2)
        )

        # 4. Learnable Scale Parameter (Start small)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Initialize
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

    def forward(self, x):
        # Save input for residual
        shortcut = x

        # Amplitude and Phase
        amp = self.amp_conv(x)
        phase = self.phase_conv(x)
        phase = torch.tanh(phase) * math.pi
        # --- ablation switch (PWAM_MODE=full|amp|phase), added for the revision ---
        import os as _os
        _mode = _os.environ.get('PWAM_MODE', 'full')
        if _mode == 'amp':      # amplitude only: phase fixed to zero
            phase = torch.zeros_like(phase)
        elif _mode == 'phase':  # phase only: amplitude fixed to one
            amp = torch.ones_like(amp)

        # Euler's Formula
        real_part = amp * torch.cos(phase)
        imag_part = amp * torch.sin(phase)

        # Reshape for Matrix Mult: (B, C, H, W) -> (B, H, W, C)
        real_part = real_part.permute(0, 2, 3, 1)
        imag_part = imag_part.permute(0, 2, 3, 1)

        # Complex Multiplication
        # (Real * Real_W) + (Imag * Imag_W)
        y_out = (real_part @ self.weight_real.t()) + (imag_part @ self.weight_imag.t())

        # Reshape back: (B, H, W, C) -> (B, C, H, W)
        y_out = y_out.permute(0, 3, 1, 2)

        # Project
        y_out = self.proj(y_out)

        # 5. CRITICAL: Residual Connection with Scale
        # If c1 == c2, add input. If not, just return y_out
        if shortcut.shape == y_out.shape:
            return shortcut + (self.gamma * y_out)
        else:
            return y_out



class PWAM_Lite(nn.Module):
    """
    PWAM_Lite (Residual Version).
    Same logic, but safer gradient flow.
    """

    def __init__(self, c1, c2=None):
        super().__init__()
        self.c2 = c2 if c2 else c1

        # 1. Channel Attention
        c_hidden = max(c1 // 16, 16)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1, c_hidden, 1, 1),
            nn.SiLU(),  # IMPROVEMENT: Switched ReLU to SiLU (Standard in YOLOv11)
            Conv(c_hidden, c1, 1, 1),
            nn.Sigmoid()
        )

        # 2. Spatial Attention
        self.spatial_att = nn.Sequential(
            Conv(c1, c1, k=7, s=1, g=c1, act=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()
        )

        # Projection
        self.proj = Conv(c1, self.c2, 1, 1) if c1 != self.c2 else nn.Identity()

    def forward(self, x):
        # Calculate Attention Maps
        c_att = self.channel_att(x)
        s_att = self.spatial_att(x)

        # IMPROVEMENT: Residual Connection
        # Instead of replacing x, we ADD the refined features to x.
        # This prevents "signal vanishing" on small objects.
        refined = x * c_att * s_att
        out = x + refined

        return self.proj(out)



class SRA(nn.Module):
    """Space-Reduction Attention (Helper for SRAW_TM)."""

    def __init__(self, dim, num_heads=8, reduction_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.reduction_ratio = reduction_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        # Spatial Reduction
        self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Query: [B, N, C]
        q = x.flatten(2).transpose(1, 2)
        q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Key/Value with Spatial Reduction
        # x_sr: [B, C, H/R, W/R]
        x_sr = self.sr(x).flatten(2).transpose(1, 2)
        x_sr = self.norm(x_sr)

        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)

        # Reshape back to [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x



class SRAW_TM(nn.Module):
    def __init__(self, c1, c2, stride=1, scales=None):
        super().__init__()
        self.stride = stride

        # 1. Input Convolution (Handle channel change or downsample)
        self.cv1 = Conv(c1, c2, 1, 1) if stride == 1 else Conv(c1, c2, 3, stride)

        # 2. The Attention Branch
        self.pwam = PWAM(c2, c2)

        # 3. The Spatial Branch (SRA)
        # Assuming SRA takes (c, c) arguments.
        self.sra = SRA(c2, num_heads=4, reduction_ratio=2)

        # 4. Fusion
        self.cv2 = Conv(c2, c2, 1, 1)

        # Shortcut condition
        self.add = stride == 1 and c1 == c2

    def forward(self, x):
        # Input processing
        y = self.cv1(x)

        # Apply PWAM (Channel/Phase attention)
        y_pwam = self.pwam(y)

        # Apply SRA (Spatial attention) on the refined features
        y_sra = self.sra(y_pwam)

        # Final fusion
        out = self.cv2(y_sra)

        # Residual connection
        return (x + out) if self.add else out



class SRAW_TM_Lite(nn.Module):
    """
    Sparse Recursive Attention (Lite Version).
    Wraps the heavy SRA attention in a Bottleneck to reduce parameters.
    """

    def __init__(self, c1, c2=None, num_heads=8, reduction=2):
        super().__init__()
        self.c2 = c2 if c2 else c1
        # Reduce internal channels by half to save size
        self.c_hidden = c1 // 2

        # Input Bottleneck (Squeeze)
        self.cv1 = Conv(c1, self.c_hidden, 1, 1)

        # The Heavy Attention Block (Running on reduced channels)
        self.attn = SRA(self.c_hidden, num_heads=num_heads, reduction_ratio=reduction)

        # Output Bottleneck (Expand)
        self.cv2 = Conv(self.c_hidden, self.c2, 1, 1)

        # Shortcut connection
        self.add = (c1 == self.c2)

    def forward(self, x):
        y = self.cv1(x)
        y = self.attn(y)
        y = self.cv2(y)
        return x + y if self.add else y



# ---------------------------------------------------------------------------
# Pretrained-compatible PWST insertions (revision): keep the layer index and the
# state-dict keys (conv/bn) of the wrapped Conv, add a zero-initialised gated branch.
# ---------------------------------------------------------------------------
class ConvPWAM(Conv):
    """Conv followed by a gated PWAM (amplitude/phase). gamma=0 at start => identical to Conv."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.pw = PWAM(c2, c2)

    def forward(self, x):
        return self.pw(super().forward(x))

    def forward_fuse(self, x):
        # Ultralytics' fuse() replaces forward by forward_fuse for every Conv subclass; without this override the PWAM branch is dropped
        return self.pw(super().forward_fuse(x))



class ConvSRAW(Conv):
    """Down-sampling Conv followed by PWAM and space-reduction attention, both on residual, gated paths."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True, heads=4, red=2):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.pw = PWAM(c2, c2)
        self.sra = SRA(c2, num_heads=heads, reduction_ratio=red)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.pw(super().forward(x))
        return y + self.gamma * self.sra(y)

    def forward_fuse(self, x):
        y = self.pw(super().forward_fuse(x))
        return y + self.gamma * self.sra(y)



class ConvSRA(Conv):
    """Down-sampling Conv followed by a gated SRA only (no PWAM)."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True, heads=4, red=2):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.sra = SRA(c2, num_heads=heads, reduction_ratio=red)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = super().forward(x)
        return y + self.gamma * self.sra(y)

    def forward_fuse(self, x):
        y = super().forward_fuse(x)
        return y + self.gamma * self.sra(y)



# ---------------------------------------------------------------------------
# Revision (v2 of the method): "never-worse-at-init" PWST blocks. They subclass C3k2 so that the pretrained
# state-dict keys are untouched, and append a zero-initialised (gamma=0) PWAM [+ SRA] residual branch.
# ---------------------------------------------------------------------------
class C3k2PW(C3k2):
    """C3k2 followed by a gated PWAM branch (phase-aware amplitude/phase aggregation)."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.pw = PWAM(c2, c2)

    def forward(self, x):
        return self.pw(super().forward(x))



class C3k2PWSRA(C3k2):
    """C3k2 followed by gated PWAM and gated space-reduction attention (SRAW-like, but position preserved)."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.pw = PWAM(c2, c2)
        self.sra = SRA(c2, num_heads=4, reduction_ratio=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.pw(super().forward(x))
        return y + self.gamma * self.sra(y)



# ---------------------------------------------------------------------------
# Revision: WaveMLP phase-aware token mixing (PATM; Tang et al., "An Image Patch is a Wave: Phase-Aware Vision MLP", CVPR 2022),
# layout of the official implementation: amplitude branches fc_h / fc_w, phase branches theta_h / theta_w, Euler expansion into
# cosine / sine parts, token mixing by a 7-tap depth-wise convolution along the height / the width, a channel branch fc_c, a
# softmax re-weighting of the three branches and an output projection.  Used here as the direct counterpart of PWAM at the same
# positions (zero-gated residual, so that every model computes the function of YOLOv11n at initialisation).
# ---------------------------------------------------------------------------
class WavePATM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=False)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=False)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=False)
        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 3), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(3, 0), groups=dim, bias=False)
        self.reweight = nn.Sequential(nn.Conv2d(dim, max(dim // 4, 8), 1), nn.GELU(), nn.Conv2d(max(dim // 4, 8), dim * 3, 1))
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):
        B, C, H, W = x.shape
        th, tw = self.theta_h_conv(x), self.theta_w_conv(x)
        xh, xw = self.fc_h(x), self.fc_w(x)
        xh = torch.cat([xh * torch.cos(th), xh * torch.sin(th)], dim=1)
        xw = torch.cat([xw * torch.cos(tw), xw * torch.sin(tw)], dim=1)
        h, w, c = self.tfc_h(xh), self.tfc_w(xw), self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        return self.proj(h * a[0] + w * a[1] + c * a[2])



class ConvWPATM(Conv):
    """Conv followed by a zero-gated WaveMLP PATM block (direct counterpart of ConvPWAM)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.wp = WavePATM(c2)
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = super().forward(x)
        return y + self.gate_bias * self.wp(y)

    def forward_fuse(self, x):
        y = super().forward_fuse(x)
        return y + self.gate_bias * self.wp(y)



# ---------------------------------------------------------------------------
# Revision (v3): domain-generalisation / photometric modules.  All of them subclass Conv or C3k2, so the pretrained state-dict
# keys are untouched, and all are the identity at inference (AmpMix, MixStyle) or at initialisation (GatedIN, UnsharpGate, FeatGIN, SpecGain, ConvLK).
# Every Conv subclass overrides forward_fuse (Ultralytics' fuse() would otherwise drop the extra branch, see ConvPWAM above).
# Learnable gates are 1-D (or 2-D) tensors whose name contains "bias": Ultralytics puts such parameters into the no-weight-decay SGD group
# (not into the Muon group, which only accepts 2-D matrices and 4-D kernels).
# ---------------------------------------------------------------------------
class AmpMix(nn.Module):
    """Phase-preserving Fourier amplitude mixing (training only).

    The amplitude spectrum of every sample is mixed with that of another sample of the batch, x' = IFFT((1-l)|X_i| + l|X_j|, angle X_i)
    (phase = structure is kept, amplitude = style/illumination/turbidity/sharpness is perturbed); image geometry and boxes are unchanged."""

    def __init__(self, p=0.5, lam_max=1.0):
        super().__init__()
        self.p, self.lam_max = p, lam_max

    def forward(self, x):
        if not self.training or x.shape[0] < 2:
            return x
        B, dt = x.shape[0], x.dtype
        f = torch.fft.rfft2(x.float(), norm="ortho")
        amp, pha = f.abs(), f.angle()
        perm = torch.randperm(B, device=x.device)
        lam = torch.rand(B, 1, 1, 1, device=x.device) * self.lam_max
        lam = lam * (torch.rand(B, 1, 1, 1, device=x.device) < self.p).float()
        amp = (1 - lam) * amp + lam * amp[perm]
        out = torch.fft.irfft2(torch.polar(amp, pha), s=x.shape[-2:], norm="ortho")
        return out.clamp_(0, 1).to(dt)



class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021): mixes the per-sample channel statistics (mean, std) of a feature map with those of another
    sample of the batch (training only, probability p, lambda ~ Beta(alpha, alpha)); parameter-free, identity at inference."""

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or x.shape[0] < 2 or torch.rand(1).item() > self.p:
            return x
        B, dt = x.shape[0], x.dtype
        xf = x.float()
        mu = xf.mean(dim=(2, 3), keepdim=True)
        sig = (xf.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        xn = (xf - mu) / sig
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        out = xn * (sig * lam + sig[perm] * (1 - lam)) + (mu * lam + mu[perm] * (1 - lam))
        return out.to(dt)



class ConvAmp0(Conv):
    """Conv whose input passes through AmpMix (training only)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.am = AmpMix()

    def forward(self, x):
        return super().forward(self.am(x))

    def forward_fuse(self, x):
        return super().forward_fuse(self.am(x))



class C3k2MS(C3k2):
    """C3k2 followed by MixStyle (training only)."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.ms = MixStyle()

    def forward(self, x):
        return self.ms(super().forward(x))

