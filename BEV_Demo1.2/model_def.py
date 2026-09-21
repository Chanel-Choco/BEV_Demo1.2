"""
model_def.py -- architecture copied verbatim from Notebook 4 (v14, LFAB v4), so the state_dicts saved by
Notebooks 2 and 3 load without any key mismatch.

If you ever change LFAB / ProposedAIDetector in the notebooks, re-copy them here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def make_radial_band_masks(H, Wf, num_bands=6, device='cpu'):
    """Non-overlapping radial bands over the (H, Wf) rFFT grid.
    (v3 used <= on both band edges, so 2 bins at 56x56 sat in two bands and got
    the sum of both gains.) Now used for REPORTING and for the SHAP-over-bands check."""
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(Wf), indexing='ij')
    yy = torch.minimum(yy, H - yy)
    radius = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    band = (radius / radius.max() * num_bands).long().clamp(max=num_bands - 1)
    return [(band == i).to(device) for i in range(num_bands)]


class LFAB(nn.Module):
    """Learnable Frequency Attention Block (v4).

    Changes vs v3, each tied to what the Notebook 4 diagnostics showed:
      * One learnable gain PER FREQUENCY BIN (per channel by default) instead of 6 radial
        bands. Generator artifacts (upsampling grids, periodic peaks) are narrow and
        directional; a radial band averages them away.
      * Gain = 2*sigmoid(LOGIT_SCALE * z), z initialised to 0 -> gain 1.0 everywhere (a neutral
        filter), range (0, 2), so the block can amplify as well as damp. v3's sigmoid capped gain at 1.
      * LOGIT_SCALE = 10: Adam moves a parameter by about `lr` per step. With ~27 optimizer steps per
        epoch, a raw logit can only travel ~0.1-0.3 in a whole run, which is why v3's band gains stayed
        at 0.49-0.51. The scale lets the gains cover their full (0, 2) range in that step budget.
      * The filter is stored only for rows 0..H//2 and mirrored, so it is a real,
        even (zero-phase) filter. It has ~13k params at 16x56x56.
      * The per-image gate is removed: in v3 it acted as a constant (Model A ~0.04, Model B
        saturated near its +/-1 bound for every source) and duplicated band_weights.
      * Still: residual, alpha-gated, FFT in fp32 (56 is not a power of 2 -> no fp16 FFT).
    """
    ALPHA_INIT = 0.1
    MAX_GAIN = 2.0
    LOGIT_SCALE = 10.0

    def __init__(self, channels, spatial_size, num_bands=6, groups=None):
        super().__init__()
        H, W = int(spatial_size[0]), int(spatial_size[1])
        groups = channels if groups is None else groups   # groups=channels -> one filter per channel
        assert channels % groups == 0, "channels must be divisible by groups"
        self.channels, self.groups, self.num_bands = channels, groups, num_bands
        self.H, self.W = H, W

        # z = 0 -> gain = MAX_GAIN * sigmoid(0) = 1.0 (neutral)
        self.gain_logits = nn.Parameter(torch.zeros(groups, H // 2 + 1, W // 2 + 1))
        self.alpha = nn.Parameter(torch.tensor(self.ALPHA_INIT))
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _gain_half(self, H, W):
        """(groups, H//2+1, W//2+1) gain for the stored (non-mirrored) half of the spectrum."""
        logits = self.gain_logits
        Hh, Wf = H // 2 + 1, W // 2 + 1
        if logits.shape[-2:] != (Hh, Wf):          # different input size -> resample the learned filter
            logits = F.interpolate(logits.unsqueeze(0), size=(Hh, Wf),
                                   mode='bilinear', align_corners=True).squeeze(0)
        return self.MAX_GAIN * torch.sigmoid(self.LOGIT_SCALE * logits)

    def _gain(self, H, W):
        """(groups, H, W//2+1) gain map for an H x W feature map (rows mirrored)."""
        gain = self._gain_half(H, W)
        k = torch.arange(H, device=gain.device)
        rows = torch.minimum(k, H - k)             # mirror rows H//2+1..H-1 onto H//2-1..1
        return gain[:, rows, :]

    def filter_features(self, x):
        B, C, H, W = x.shape
        spec = torch.fft.rfft2(x.float(), norm="ortho")
        gain = self._gain(H, W).repeat_interleave(C // self.groups, dim=0)   # (C, H, Wf)
        return torch.fft.irfft2(spec * gain.unsqueeze(0), s=(H, W), norm="ortho")

    def forward(self, x):
        return x + self.alpha * self.fusion(self.filter_features(x).to(x.dtype))

    # ---- reporting helpers ----
    @torch.no_grad()
    def gain_map(self):
        """(H, W//2+1) learned gain, averaged over channels/groups. Plot it with imshow:
        a ring means 'radial' behaviour, streaks along an axis mean it found a directional artifact."""
        return self._gain(self.H, self.W).mean(0)

    @torch.no_grad()
    def band_gains(self):
        """Mean gain in each of the num_bands radial bands (low -> high freq), for the summary tables."""
        g = self.gain_map()
        masks = make_radial_band_masks(self.H, self.W // 2 + 1, self.num_bands, g.device)
        return torch.stack([g[m].mean() for m in masks])


class ProposedAIDetector(nn.Module):
    # features[1] of MobileNetV3-Small and features[2] of EfficientNet-B0 both output a 56x56 map.
    LFAB_BLOCK_DEFAULTS = {'mobilenet': 1, 'efficientnet': 2}

    def __init__(self, embedding_dim=128, pretrained=False, backbone_type='mobilenet', dropout_p=0.2, use_lfab=True,
                 lfab_after_block=None):
        super(ProposedAIDetector, self).__init__()
        self.backbone_type = backbone_type
        self.use_lfab = use_lfab
        # LFAB v4: the block is applied after backbone.features[lfab_after_block], on a 56x56 map at
        # 224px input (V11 used the final 7x7 map, V12 a 28x28 map). None -> per-backbone default below.

        if backbone_type == 'mobilenet':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights).features
            self.in_channels = 576
        elif backbone_type == 'efficientnet':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights).features
            self.in_channels = 1280
        else:
            raise ValueError("Unsupported backbone. Use 'mobilenet' or 'efficientnet'.")

        self.lfab_after_block = self.LFAB_BLOCK_DEFAULTS[backbone_type] if lfab_after_block is None else lfab_after_block

        if pretrained and weights is not None:
            print(f"[ProposedAIDetector] Loaded ImageNet-pretrained weights for '{backbone_type}' backbone: {weights}")
        else:
            print(f"[ProposedAIDetector] '{backbone_type}' backbone initialized with RANDOM weights (pretrained=False).")

        # Probe the feature-map shape at the LFAB insertion point (eval mode + no_grad so BatchNorm
        # running statistics are not touched), so LFAB is always built with the right channel count.
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            probe = self.backbone[:self.lfab_after_block + 1](torch.zeros(1, 3, 224, 224))
        self.backbone.train(was_training)
        self.lfab_feature_shape = tuple(probe.shape[1:])   # (C, H, W)

        self.lfab = LFAB(channels=self.lfab_feature_shape[0], spatial_size=self.lfab_feature_shape[1:])
        self.pool = nn.AdaptiveAvgPool2d(1)

        if backbone_type == 'mobilenet':
            self.embedding_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.in_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Linear(256, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.embedding_head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_p),
                nn.Linear(self.in_channels, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True)
            )

        self.final_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # Run the backbone block by block so LFAB can sit in the middle of it.
        for i, block in enumerate(self.backbone):
            x = block(x)
            if self.use_lfab and i == self.lfab_after_block:
                x = self.lfab(x)

        pooled = self.pool(x)
        raw_embed = self.embedding_head(pooled)
        norm_embed = F.normalize(raw_embed, p=2, dim=1)
        output = self.final_head(raw_embed)   # classify on the UN-normalized embedding — norm_embed
                                                # stays reserved for the contrastive loss only, so the
                                                # classifier is no longer capped to what fits on a unit sphere

        return output, norm_embed
