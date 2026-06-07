# -*- coding: utf-8 -*-
"""
ConvNeXt + Mamba Medical Image Classification

Input : (B,3,224,224)
Output: (B,num_classes)
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Dict, Any, List
from config import AblationConfig

warnings.filterwarnings("ignore")


class ReliabilityGuidedBilateralFusionGate(nn.Module):
    """
    Reliability-guided bilateral fusion gate.

    Computes per-sample branch weights from confidence-derived reliability cues:
      - predictive entropy (conv & mamba branches)
      - maximum class probability (conv & mamba branches)
      - classification margin = top1 - top2 (conv & mamba branches)

    Additional embedding-level reliability cues (not treated as uncertainty estimates):
      - embedding L2 norm (conv & mamba)
      - cosine similarity between branch embeddings
      - absolute entropy difference between branches

    Output: alpha_conv, alpha_mamba with alpha_conv + alpha_mamba = 1 (softmax-normalized).
    """
    def __init__(self, in_dim: int = 10, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, 2)  # logits for [conv, mamba]
        )

    @staticmethod
    def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    def forward(
        self,
        conv_logits: torch.Tensor,
        mamba_logits: torch.Tensor,
        conv_emb: torch.Tensor,
        mamba_emb: torch.Tensor,
        temperature: float = 1.0,
        detach_features: bool = False,
        gate_min: float = 0.05,
    ):
        # prob stats
        pc = F.softmax(conv_logits, dim=-1)
        pm = F.softmax(mamba_logits, dim=-1)

        ent_c = self._entropy(pc)
        ent_m = self._entropy(pm)

        maxp_c, _ = pc.max(dim=-1)
        maxp_m, _ = pm.max(dim=-1)

        # margin = top1 - top2
        top2_c = pc.topk(2, dim=-1).values
        top2_m = pm.topk(2, dim=-1).values
        margin_c = top2_c[:, 0] - top2_c[:, 1]
        margin_m = top2_m[:, 0] - top2_m[:, 1]

        # emb stats
        c_norm = conv_emb.norm(dim=-1)
        m_norm = mamba_emb.norm(dim=-1)

        cos = F.cosine_similarity(conv_emb, mamba_emb, dim=-1)

        feats = torch.stack([
            ent_c, ent_m,
            maxp_c, maxp_m,
            margin_c, margin_m,
            c_norm, m_norm,
            cos,
            (ent_c - ent_m).abs(),
        ], dim=-1)

        if detach_features:
            feats = feats.detach()

        gate_logits = self.net(feats) / max(temperature, 1e-6)
        w = F.softmax(gate_logits, dim=-1)  # (B,2)
        w_c = w[:, 0].clamp(min=gate_min, max=1.0 - gate_min)
        w_m = (1.0 - w_c)
        return w_c, w_m


# Legacy alias for backward compatibility (deprecated — use ReliabilityGuidedBilateralFusionGate)
UGBFReliabilityGate = ReliabilityGuidedBilateralFusionGate


# =========================================================
# 0) Utilities
# =========================================================
def softmax_entropy(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy of softmax distribution. Returns shape (B,)"""
    p = F.softmax(logits, dim=-1).clamp_min(eps)
    ent = -(p * p.log()).sum(dim=-1)
    return ent


def entropy_to_confidence(ent: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Convert entropy to confidence score in (0, +inf):
    conf = exp(-ent/T). Higher entropy -> lower confidence
    """
    return torch.exp(-ent / max(temperature, 1e-6))


def normalize_two_weights(w1: torch.Tensor, w2: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    s = (w1 + w2).clamp_min(eps)
    return w1 / s, w2 / s


class MLPHead(nn.Module):
    """Simple MLP head: in -> hidden -> hidden/2 -> out"""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================
# 1) ConvNeXt Backbone Loader (single & multi-scale)
# =========================================================
def _build_convnext_backbone(
    model_name: str = "convnext_base",
    pretrained: bool = True,
    in_channels: int = 3,
    out_indices: Tuple[int, ...] = (3,),
):
    """
    Returns a backbone that can output feature maps.
    Priority: timm features_only -> torchvision fallback (single-scale only).

    - If timm is available: backbone(x) returns List[Tensor] for requested out_indices.
    - If torchvision fallback: backbone is tv_model.features (Sequential) and only supports last-stage output.
    """
    # Try timm first
    try:
        import timm  # type: ignore

        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=out_indices,
        )
        backbone._is_timm_features_only = True
        backbone._out_indices = out_indices
        return backbone
    except Exception as e_timm:
        # Fallback to torchvision
        try:
            from torchvision.models import convnext_base, ConvNeXt_Base_Weights  # type: ignore

            if model_name != "convnext_base":
                raise ValueError(
                    f"torchvision fallback only implemented for 'convnext_base', got {model_name}"
                )

            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            tv_model = convnext_base(weights=weights)

            # Patch first conv for in_channels != 3
            original_first_conv = tv_model.features[0][0]
            tv_model.features[0][0] = nn.Conv2d(
                in_channels,
                original_first_conv.out_channels,
                kernel_size=original_first_conv.kernel_size,
                stride=original_first_conv.stride,
                padding=original_first_conv.padding,
                bias=False,
            )
            if in_channels != 3:
                nn.init.kaiming_normal_(tv_model.features[0][0].weight, mode="fan_out", nonlinearity="relu")

            backbone = tv_model.features
            backbone._is_timm_features_only = False
            backbone._out_indices = (3,)  # only last-stage
            return backbone
        except Exception as e_tv:
            raise RuntimeError(
                "Failed to build ConvNeXt backbone. "
                f"timm error: {repr(e_timm)}; torchvision error: {repr(e_tv)}"
            )


def _forward_convnext_backbone(backbone, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Returns a list of feature maps.
    - timm: already multi-stage list
    - torchvision fallback: returns [last_stage]
    """
    if getattr(backbone, "_is_timm_features_only", False):
        feats = backbone(x)  # List[Tensor]
        return feats
    else:
        # torchvision: only last-stage map
        return [backbone(x)]


# =========================================================
# 2) Mamba vision encoder (single & multi-layer readout)
# =========================================================
class MambaLayer(nn.Module):
    """Mamba layer adapted for (B, N, C) tokens."""
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        try:
            from mamba_ssm import Mamba  # type: ignore
        except Exception as e:
            raise RuntimeError("mamba_ssm is not available. Please install mamba-ssm.") from e
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = x + residual
        x = self.act(x)
        return x


class MambaVisionEncoder(nn.Module):
    """
    Patchify image -> tokens -> Mamba blocks
    Outputs:
      - embedding (B,C) via token mean of final layer
      - optional multi-layer embeddings (B,C) list via readout_layers
    """
    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        readout_layers: Optional[Tuple[int, ...]] = None,  # e.g. (1,3) 0-based indices
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.readout_layers = readout_layers

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        base_grid = img_size // patch_size
        base_num_patches = base_grid * base_grid
        self.pos_embed = nn.Parameter(torch.zeros(1, base_num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [MambaLayer(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def _interpolate_pos_embed(self, x_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = x_tokens.shape
        if self.pos_embed.shape[1] == N:
            return self.pos_embed

        old_N = self.pos_embed.shape[1]
        old_size = int(math.sqrt(old_N))
        new_size = int(math.sqrt(N))
        if old_size * old_size != old_N or new_size * new_size != N:
            return F.interpolate(
                self.pos_embed.transpose(1, 2), size=N, mode="linear", align_corners=False
            ).transpose(1, 2)

        pos = self.pos_embed.reshape(1, old_size, old_size, C).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(new_size, new_size), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, N, C)
        return pos

    # dual_model.py
    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        """
        Returns:
          - emb: (B, C)
          - if return_tokens:
              token_map: (B, C, H', W')
              grid_hw: (H', W')
        """
        param_device = next(self.parameters()).device
        if x.device != param_device:
            x = x.to(param_device, non_blocking=True)

        # patchify
        x = self.patch_embed(x)  # (B,C,H',W')
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B,N,C)

        # pos embed
        pos = self._interpolate_pos_embed(tokens)
        if pos.device != tokens.device:
            pos = pos.to(tokens.device, non_blocking=True)

        tokens = tokens + pos

        for blk in self.blocks:
            tokens = self.drop(blk(tokens))

        tokens = self.norm(tokens)  # (B,N,C)
        emb = tokens.mean(dim=1)  # (B,C)

        if return_tokens:
            token_map = tokens.transpose(1, 2).reshape(B, C, H, W)
            return emb, token_map, (H, W)
        return emb


class SAFFusion(nn.Module):
    """
    SAF: Edge-Pyramid + Bi-directional Cross-Attention + Dual (Spatial/Channel) Attention + DW-Separable Fusion
      forward(conv_map, mamba_map, edge_att) -> (conv_emb, mamba_emb, fused_emb)
    """
    def __init__(
        self,
        conv_in: int,
        mamba_in: int,
        saf_dim: int = 256,
        attn_heads: int = 8,
        token_hw: int = 14,
        pyramid_scales: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        fuse: str = "cat",
    ):
        super().__init__()
        self.saf_dim = saf_dim
        self.token_hw = token_hw
        self.pyramid_scales = pyramid_scales
        self.fuse = fuse

        # 1) saf_dim
        self.conv_proj = nn.Sequential(
            nn.Conv2d(conv_in, saf_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(saf_dim),
            nn.ReLU(inplace=True),
        )
        self.mamba_proj = nn.Sequential(
            nn.Conv2d(mamba_in, saf_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(saf_dim),
            nn.ReLU(inplace=True),
        )

        # 2) Edge Pyramid encoder
        pyr_blocks = []
        for s in pyramid_scales:
            # 1x1 -> 3x3 -> 3x3
            pyr_blocks.append(nn.Sequential(
                nn.Conv2d(1, saf_dim // 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(saf_dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(saf_dim // 4, saf_dim // 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(saf_dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(saf_dim // 4, saf_dim // 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(saf_dim // 4),
                nn.ReLU(inplace=True),
            ))
        self.edge_pyr = nn.ModuleList(pyr_blocks)

        # 3) Channel Attention（SE-like）
        hidden = max(16, saf_dim // 8)
        self.chan_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(saf_dim, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, saf_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # 4) Spatial Attention
        self.spa_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # 5) Bi-directional Cross Attention
        self.qkv_norm = nn.LayerNorm(saf_dim)
        self.attn_c2m = nn.MultiheadAttention(embed_dim=saf_dim, num_heads=attn_heads, dropout=dropout, batch_first=True)
        self.attn_m2c = nn.MultiheadAttention(embed_dim=saf_dim, num_heads=attn_heads, dropout=dropout, batch_first=True)

        self.attn_ffn = nn.Sequential(
            nn.Linear(saf_dim, saf_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(saf_dim * 4, saf_dim),
        )
        self.attn_drop = nn.Dropout(dropout)

        # DW-Separable Fusion block
        # concat(conv, mamba, edge_context)
        in_fuse = saf_dim * 2 + (saf_dim // 4)
        self.fuse_reduce = nn.Sequential(
            nn.Conv2d(in_fuse, saf_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(saf_dim),
            nn.ReLU(inplace=True),
        )
        self.dwsep_fuse = nn.Sequential(
            # depthwise
            nn.Conv2d(saf_dim, saf_dim, kernel_size=3, padding=1, groups=saf_dim, bias=False),
            nn.BatchNorm2d(saf_dim),
            nn.ReLU(inplace=True),
            # pointwise
            nn.Conv2d(saf_dim, saf_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(saf_dim),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=dropout),

            
            nn.Conv2d(saf_dim, saf_dim, kernel_size=3, padding=1, groups=saf_dim, bias=False),
            nn.BatchNorm2d(saf_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(saf_dim, saf_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(saf_dim),
            nn.ReLU(inplace=True),
        )

        # 7) refinement
        self.refine = nn.Sequential(
            nn.Conv2d(saf_dim, saf_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(saf_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(saf_dim, saf_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(saf_dim),
        )
        self.refine_act = nn.ReLU(inplace=True)

    @staticmethod
    def _resize_att(att: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(att, size=hw, mode="bilinear", align_corners=False)

    def _spatial_att(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        a = self.spa_att(torch.cat([avg, mx], dim=1))
        return a

    def _apply_dual_att(self, x: torch.Tensor) -> torch.Tensor:
        # channel
        ca = self.chan_att(x)
        x = x * ca
        # spatial
        sa = self._spatial_att(x)
        x = x * sa
        return x

    def _to_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: (B,D,H,W) -> pool to (B,D,T,T) -> tokens (B,N,D)
        x2 = F.adaptive_avg_pool2d(x, (self.token_hw, self.token_hw))
        B, D, Ht, Wt = x2.shape
        tok = x2.flatten(2).transpose(1, 2)  # (B,N,D)
        return tok, (Ht, Wt)

    def _from_tokens(self, tok: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        # tok: (B,N,D) -> (B,D,H,W)
        B, N, D = tok.shape
        Ht, Wt = hw
        return tok.transpose(1, 2).reshape(B, D, Ht, Wt)

    def _cross_attn(self, conv_f: torch.Tensor, mamba_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # conv_f/mamba_f: (B,D,H,W)
        c_tok, chw = self._to_tokens(conv_f)
        m_tok, mhw = self._to_tokens(mamba_f)

        # LN
        c_tok_n = self.qkv_norm(c_tok)
        m_tok_n = self.qkv_norm(m_tok)

        # c attends m
        c2m, _ = self.attn_c2m(query=c_tok_n, key=m_tok_n, value=m_tok_n, need_weights=False)
        c_tok = c_tok + self.attn_drop(c2m)
        c_tok = c_tok + self.attn_drop(self.attn_ffn(self.qkv_norm(c_tok)))

        # m attends c
        m2c, _ = self.attn_m2c(query=m_tok_n, key=c_tok_n, value=c_tok_n, need_weights=False)
        m_tok = m_tok + self.attn_drop(m2c)
        m_tok = m_tok + self.attn_drop(self.attn_ffn(self.qkv_norm(m_tok)))

        c_map = self._from_tokens(c_tok, chw)
        m_map = self._from_tokens(m_tok, mhw)
        return c_map, m_map

    def forward(self, conv_map: torch.Tensor, mamba_map: torch.Tensor, edge_att: torch.Tensor):
        """
        conv_map:  (B,Cc,Hc,Wc)
        mamba_map: (B,Cm,Hm,Wm)
        edge_att:  (B,1,Himg,Wimg)
        """
        # 1)
        att_c = self._resize_att(edge_att, conv_map.shape[-2:])
        att_m = self._resize_att(edge_att, mamba_map.shape[-2:])

        conv = self.conv_proj(conv_map * att_c)        # (B,D,Hc,Wc)
        mamba = self.mamba_proj(mamba_map * att_m)     # (B,D,Hm,Wm)
        mamba = F.interpolate(mamba, size=conv.shape[-2:], mode="bilinear", align_corners=False)

        # 2)
        conv = self._apply_dual_att(conv)
        mamba = self._apply_dual_att(mamba)

        # 3) Cross-Attention
        conv_ca, mamba_ca = self._cross_attn(conv, mamba)
        conv = conv + F.interpolate(conv_ca, size=conv.shape[-2:], mode="bilinear", align_corners=False)
        mamba = mamba + F.interpolate(mamba_ca, size=conv.shape[-2:], mode="bilinear", align_corners=False)

        # 4) Edge Pyramid context
        edge_ctx_sum = 0.0
        for blk, s in zip(self.edge_pyr, self.pyramid_scales):
            e = edge_att
            if s > 1:
                e = F.avg_pool2d(e, kernel_size=s, stride=s, ceil_mode=True)
            e = blk(e)
            e = F.interpolate(e, size=conv.shape[-2:], mode="bilinear", align_corners=False)
            edge_ctx_sum = edge_ctx_sum + e
        edge_ctx = edge_ctx_sum / float(len(self.pyramid_scales))  # (B, D//4, H, W)

        # 5) concat(conv, mamba, edge_ctx) -> reduce -> dwsep -> refine(res)
        fused = torch.cat([conv, mamba, edge_ctx], dim=1)
        fused = self.fuse_reduce(fused)
        fused = self.dwsep_fuse(fused)

        fused = self.refine_act(fused + self.refine(fused))

        # 6) embedding
        conv_emb = F.adaptive_avg_pool2d(conv, (1, 1)).flatten(1)      # (B,D)
        mamba_emb = F.adaptive_avg_pool2d(mamba, (1, 1)).flatten(1)    # (B,D)
        fused_emb = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)    # (B,D)

        return conv_emb, mamba_emb, fused_emb


# =========================================================
# 5) Three switchable networks (baseline kept)
# =========================================================
class ConvNeXtNet(nn.Module):
    """ConvNeXt-only classifier."""
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        backbone_name: str = "convnext_base",
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = _build_convnext_backbone(
            model_name=backbone_name, pretrained=pretrained, in_channels=in_channels, out_indices=(3,)
        )
        self.backbone_out_dim = 1024  # convnext_base last stage
        self.feature_dim = self.backbone_out_dim
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B,1024,H,W)->GAP->(B,1024)"""
        feat_map = _forward_convnext_backbone(self.backbone, x)[0]
        emb = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = _forward_convnext_backbone(self.backbone, x)[0]  # (B,1024,H,W)
        return self.head(feat_map)


class MambaNet(nn.Module):
    """Mamba-only classifier."""
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = MambaVisionEncoder(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.feature_dim = embed_dim
        self.classifier = MLPHead(embed_dim, num_classes, hidden_dim=embed_dim, dropout=dropout)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B,embed_dim)"""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return self.classifier(emb)


# =========================================================
# 6) Dual-stream with fusion_type
# =========================================================
# dual_model.py
class DualConvNeXtMambaNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        convnext_name: str = "convnext_base",
        convnext_pretrained: bool = True,
        mamba_img_size: int = 224,
        mamba_patch_size: int = 16,
        mamba_embed_dim: int = 512,
        mamba_depth: int = 4,
        fusion_dim: int = 512,
        dropout: float = 0.2,
        ablation: Optional[AblationConfig] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ab = ablation if ablation is not None else AblationConfig()

        #
        use_saf = bool(self.ab.use_saf and (self.ab.saf_prior != "none"))
        # Backward compat: support legacy use_ugbf field if present
        _ab_use_rgbf = getattr(self.ab, "use_rgbf", None)
        if _ab_use_rgbf is not None:
            use_rgbf = bool(_ab_use_rgbf)
        else:
            use_rgbf = bool(getattr(self.ab, "use_ugbf", False))

        # ConvNeXt backbone
        self.convnext_backbone = _build_convnext_backbone(
            model_name=convnext_name,
            pretrained=convnext_pretrained,
            in_channels=in_channels,
            out_indices=(3,),
        )

        # Mamba encoder
        self.mamba_encoder = MambaVisionEncoder(
            in_channels=in_channels,
            img_size=mamba_img_size,
            patch_size=mamba_patch_size,
            embed_dim=mamba_embed_dim,
            depth=mamba_depth,
            dropout=dropout,
        )

        # dims
        self.conv_dim = 1024
        self.mamba_dim = getattr(self.mamba_encoder, "embed_dim", mamba_embed_dim)
        self.saf_dim = getattr(self.ab, "saf_dim", 256)
        self.rgbf_dim = 512

        # Branch heads (RGBF gate needs logits)
        if use_rgbf:
            self.conv_branch_head = nn.Linear(self.conv_dim, num_classes)
            self.mamba_branch_head = nn.Linear(self.mamba_dim, num_classes)
        else:
            self.conv_branch_head = None
            self.mamba_branch_head = None

        # Baseline concat classifier
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.conv_dim + self.mamba_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(512, num_classes)

        # =========================
        # SAF
        # =========================
        if use_saf:
            self.saf = SAFFusion(
                conv_in=self.conv_dim,
                mamba_in=self.mamba_dim,
                saf_dim=self.saf_dim,
                fuse=self.ab.saf_fuse,
            )
            self.saf_classifier = nn.Linear(self.saf_dim, num_classes)
        else:
            self.saf = None
            self.saf_classifier = None

        # =========================
        # RGBF (reliability-guided bilateral fusion)
        # =========================
        if use_rgbf:
            self.conv_rgbf_proj_raw = nn.Linear(self.conv_dim, self.rgbf_dim)
            self.mamba_rgbf_proj_raw = nn.Linear(self.mamba_dim, self.rgbf_dim)

            if use_saf:
                self.conv_rgbf_proj_saf = nn.Linear(self.saf_dim, self.rgbf_dim)
                self.mamba_rgbf_proj_saf = nn.Linear(self.saf_dim, self.rgbf_dim)
            else:
                self.conv_rgbf_proj_saf = None
                self.mamba_rgbf_proj_saf = None

            self.rgbf_fuse = nn.Sequential(
                nn.LayerNorm(self.rgbf_dim),
                nn.Linear(self.rgbf_dim, self.rgbf_dim),
                nn.GELU(),
            )
            self.rgbf_classifier = nn.Linear(self.rgbf_dim, num_classes)

            # Reliability-guided bilateral fusion gate (entropy + max_prob + margin)
            self.reliability_gate = ReliabilityGuidedBilateralFusionGate(
                in_dim=10, hidden=128, dropout=dropout
            )
            # Legacy alias (deprecated)
            self.ugbf_gate_net = self.reliability_gate

            self.rgbf_temperature = getattr(self.ab, "rgbf_temperature",
                                  getattr(self.ab, "ugbf_temperature", 1.0))
            self.detach_gate = getattr(self.ab, "detach_gate", False)
            self.gate_min = getattr(self.ab, "gate_min", 0.05)
        else:
            self.conv_rgbf_proj_raw = None
            self.mamba_rgbf_proj_raw = None
            self.conv_rgbf_proj_saf = None
            self.mamba_rgbf_proj_saf = None
            self.rgbf_fuse = None
            self.rgbf_classifier = None
            self.reliability_gate = None
            self.ugbf_gate_net = None

            self.rgbf_temperature = getattr(self.ab, "rgbf_temperature",
                                  getattr(self.ab, "ugbf_temperature", 1.0))
            self.detach_gate = getattr(self.ab, "detach_gate", False)
            self.gate_min = getattr(self.ab, "gate_min", 0.05)

        # feature_dim
        if use_rgbf:
            self.feature_dim = self.rgbf_dim
        elif use_saf:
            self.feature_dim = self.saf_dim
        else:
            self.feature_dim = 512

        self._apply_ablation_trainability()

    # -------------------- utilities --------------------
    @staticmethod
    def _edge_attention_map(x: torch.Tensor) -> torch.Tensor:
        """
        Build edge prior attention map (B,1,H,W) from input tensor x (already normalized is OK).
        """
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]  # (B,1,H,W)

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)  # (B,1,H,W)

        # normalize to [0,1] per-sample
        B = mag.size(0)
        mag_flat = mag.view(B, -1)
        mn = mag_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        mx = mag_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        att = (mag - mn) / (mx - mn + 1e-8)
        return att

    def _rgbf_gate(
            self,
            conv_logits: torch.Tensor,
            mamba_logits: torch.Tensor,
            conv_emb_for_gate: torch.Tensor,
            mamba_emb_for_gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RGBF: reliability-guided bilateral fusion gate.
        Returns: w_c, w_m  (B,) with w_c + w_m = 1.
        """
        w_c, w_m = self.reliability_gate(
            conv_logits=conv_logits,
            mamba_logits=mamba_logits,
            conv_emb=conv_emb_for_gate,
            mamba_emb=mamba_emb_for_gate,
            temperature=self.rgbf_temperature,
            detach_features=self.detach_gate,
            gate_min=self.gate_min,
        )
        return w_c, w_m

    # Legacy alias (deprecated)
    def _ugbf_gate(self, *args, **kwargs):
        import warnings
        warnings.warn("_ugbf_gate is deprecated. Use _rgbf_gate instead.", DeprecationWarning, stacklevel=2)
        return self._rgbf_gate(*args, **kwargs)

    # -------------------- stream encoders --------------------
    def _conv_map(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = _forward_convnext_backbone(self.convnext_backbone, x)[-1]  # (B,1024,Hc,Wc)
        return feat_map

    def _mamba_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
         mamba_emb_raw + mamba_token_map，
        """
        x_device = x.device

        mamba_device = next(self.mamba_encoder.parameters()).device
        if x.device != mamba_device:
            x_m = x.to(mamba_device, non_blocking=True)
        else:
            x_m = x

        emb, token_map, _ = self.mamba_encoder(x_m, return_tokens=True)

        if emb.device != x_device:
            emb = emb.to(x_device, non_blocking=True)
        if token_map.device != x_device:
            token_map = token_map.to(x_device, non_blocking=True)

        return emb, token_map

    def _apply_ablation_trainability(self) -> None:
        use_c, use_m = self.ab.use_convnext, self.ab.use_mamba
        use_saf = bool(self.ab.use_saf and (self.ab.saf_prior != "none"))
        use_rgbf = bool(getattr(self.ab, "use_rgbf",
                      getattr(self.ab, "use_ugbf", False)))

        def _set_trainable(mod, trainable: bool):
            if mod is None:
                return
            for p in mod.parameters():
                p.requires_grad_(trainable)

        _set_trainable(self.conv_branch_head, False)
        _set_trainable(self.mamba_branch_head, False)

        _set_trainable(self.fusion_mlp, False)
        _set_trainable(self.classifier, False)

        _set_trainable(self.saf, False)
        _set_trainable(self.saf_classifier, False)

        _set_trainable(self.conv_rgbf_proj_raw, False)
        _set_trainable(self.mamba_rgbf_proj_raw, False)
        _set_trainable(self.conv_rgbf_proj_saf, False)
        _set_trainable(self.mamba_rgbf_proj_saf, False)
        _set_trainable(self.rgbf_fuse, False)
        _set_trainable(self.rgbf_classifier, False)

        if use_c and (not use_m):
            _set_trainable(self.mamba_encoder, False)
            _set_trainable(self.conv_branch_head, True)
            return

        if use_m and (not use_c):
            _set_trainable(self.convnext_backbone, False)
            _set_trainable(self.mamba_branch_head, True)
            return

        if use_saf and use_rgbf:
            _set_trainable(self.saf, True)
            _set_trainable(self.conv_branch_head, True)
            _set_trainable(self.mamba_branch_head, True)

            _set_trainable(self.conv_rgbf_proj_raw, True)
            _set_trainable(self.mamba_rgbf_proj_raw, True)
            _set_trainable(self.conv_rgbf_proj_saf, True)
            _set_trainable(self.mamba_rgbf_proj_saf, True)

            _set_trainable(self.rgbf_fuse, True)
            _set_trainable(self.rgbf_classifier, True)
            return

        if use_saf and (not use_rgbf):
            _set_trainable(self.saf, True)
            _set_trainable(self.saf_classifier, True)
            return

        if (not use_saf) and use_rgbf:
            _set_trainable(self.conv_branch_head, True)
            _set_trainable(self.mamba_branch_head, True)

            _set_trainable(self.conv_rgbf_proj_raw, True)
            _set_trainable(self.mamba_rgbf_proj_raw, True)
            _set_trainable(self.rgbf_fuse, True)
            _set_trainable(self.rgbf_classifier, True)
            return

        # baseline concat
        _set_trainable(self.fusion_mlp, True)
        _set_trainable(self.classifier, True)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the final fused representation before final classifier (for analysis).
        """
        logits, feat = self._forward_impl(x, return_feat=True)
        return feat

    def _forward_impl(self, x: torch.Tensor, return_feat: bool = False,
                      return_aux: bool = False):
        use_c, use_m = self.ab.use_convnext, self.ab.use_mamba
        use_saf = bool(self.ab.use_saf and (self.ab.saf_prior != "none"))
        use_rgbf = bool(getattr(self.ab, "use_rgbf",
                      getattr(self.ab, "use_ugbf", False)))
        _w_c, _w_m = None, None  # for return_aux

        def _pack(logits, feat=None):
            if return_aux and return_feat:
                return logits, feat, _w_c, _w_m
            if return_aux:
                return logits, _w_c, _w_m
            if return_feat:
                return logits, feat
            return logits

        if (not use_c) and (not use_m):
            raise ValueError("Both streams are disabled.")

        # -------- single stream --------
        if use_c and (not use_m):
            conv_map = self._conv_map(x)
            conv_emb = F.adaptive_avg_pool2d(conv_map, (1, 1)).flatten(1)
            if self.conv_branch_head is None:
                raise RuntimeError(
                    "conv_branch_head is None. Please enable use_rgbf or create conv head for single-stream.")
            logits = self.conv_branch_head(conv_emb)
            return _pack(logits, feat=conv_emb)

        if use_m and (not use_c):
            mamba_emb = self.mamba_encoder(x, return_tokens=False)
            if self.mamba_branch_head is None:
                raise RuntimeError(
                    "mamba_branch_head is None. Please enable use_rgbf or create mamba head for single-stream.")
            logits = self.mamba_branch_head(mamba_emb)
            return _pack(logits, feat=mamba_emb)

        # -------- dual stream features --------
        conv_map = self._conv_map(x)
        conv_emb_raw = F.adaptive_avg_pool2d(conv_map, (1, 1)).flatten(1)

        mamba_emb_raw, mamba_map = self._mamba_tokens(x)

        # device
        if mamba_emb_raw.device != conv_emb_raw.device:
            mamba_emb_raw = mamba_emb_raw.to(conv_emb_raw.device, non_blocking=True)
        if mamba_map.device != conv_emb_raw.device:
            mamba_map = mamba_map.to(conv_emb_raw.device, non_blocking=True)

        # RGBF gate (reliability-guided bilateral fusion)
        if use_rgbf:
            if self.conv_branch_head is None or self.mamba_branch_head is None:
                raise RuntimeError("RGBF is enabled but branch heads are None (not constructed).")
            conv_logits_raw = self.conv_branch_head(conv_emb_raw)
            mamba_logits_raw = self.mamba_branch_head(mamba_emb_raw)
        else:
            conv_logits_raw = None
            mamba_logits_raw = None

        # -------- SAF path --------
        if use_saf:
            if self.saf is None or self.saf_classifier is None:
                raise RuntimeError("SAF is enabled but SAF modules are None (not constructed).")
            edge_att = self._edge_attention_map(x)
            conv_saf_emb, mamba_saf_emb, fused_emb = self.saf(conv_map, mamba_map, edge_att)

            if use_rgbf:
                _w_c, _w_m = self._rgbf_gate(conv_logits_raw, mamba_logits_raw)

                if self.conv_rgbf_proj_saf is None or self.mamba_rgbf_proj_saf is None:
                    raise RuntimeError("SAF+RGBF enabled but saf projections are None.")

                c = self.conv_rgbf_proj_saf(F.normalize(conv_saf_emb, dim=1))
                m = self.mamba_rgbf_proj_saf(F.normalize(mamba_saf_emb, dim=1))

                fused = (_w_c.unsqueeze(1) * c) + (_w_m.unsqueeze(1) * m)
                fused = self.rgbf_fuse(fused)
                logits = self.rgbf_classifier(fused)
                return _pack(logits, feat=fused)

            logits = self.saf_classifier(fused_emb)
            return _pack(logits, feat=fused_emb)

        # -------- No SAF --------
        if use_rgbf:
            _w_c, _w_m = self._rgbf_gate(conv_logits_raw, mamba_logits_raw)

            if self.conv_rgbf_proj_raw is None or self.mamba_rgbf_proj_raw is None:
                raise RuntimeError("RGBF enabled but raw projections are None.")

            c = self.conv_rgbf_proj_raw(F.normalize(conv_emb_raw, dim=1))
            m = self.mamba_rgbf_proj_raw(F.normalize(mamba_emb_raw, dim=1))

            fused = (_w_c.unsqueeze(1) * c) + (_w_m.unsqueeze(1) * m)
            fused = self.rgbf_fuse(fused)
            logits = self.rgbf_classifier(fused)
            return _pack(logits, feat=fused)

        # baseline concat
        fused = self.fusion_mlp(torch.cat([conv_emb_raw, mamba_emb_raw], dim=1))
        logits = self.classifier(fused)
        return _pack(logits, feat=fused)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        return self._forward_impl(x, return_feat=False, return_aux=return_aux)


# =========================================================
# 7) Factory wrapper
# =========================================================
class MedicalImageClassifier(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 pretrained: bool = True,
                 ablation: Optional[AblationConfig] = None,
                 fusion_type: Optional[str] = None,
                 fusion_strategy: Optional[str] = None,
                 use_bpaco: Optional[bool] = None,
                 bpaco_enabled: Optional[bool] = None,
                 in_channels: int = 3,
                 convnext_name: str = "convnext_base",
                 convnext_pretrained: bool = True,
                 mamba_img_size: int = 224,
                 mamba_patch_size: int = 16,
                 mamba_embed_dim: int = 512,
                 mamba_depth: int = 4,
                 fusion_dim: int = 512,
                 dropout: float = 0.2):
        super().__init__()
        self.model_name = str(model_name).lower().strip()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_type = self.model_name

        if self.model_name == "dual":
            self.model = DualConvNeXtMambaNet(
                num_classes=num_classes,
                in_channels=in_channels,
                convnext_name=convnext_name,
                convnext_pretrained=convnext_pretrained,
                mamba_img_size=mamba_img_size,
                mamba_patch_size=mamba_patch_size,
                mamba_embed_dim=mamba_embed_dim,
                mamba_depth=mamba_depth,
                fusion_dim=fusion_dim,
                dropout=dropout,
                ablation=ablation,
            )
        elif self.model_name == "convnext":
            self.model = ConvNeXtNet(
                num_classes=num_classes,
                in_channels=in_channels,
                backbone_name=convnext_name,
                pretrained=pretrained,
                dropout=dropout,
            )
        elif self.model_name == "mamba":
            self.model = MambaNet(
                num_classes=num_classes,
                in_channels=in_channels,
                img_size=mamba_img_size,
                patch_size=mamba_patch_size,
                embed_dim=mamba_embed_dim,
                depth=mamba_depth,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}. Use convnext/mamba/dual.")

        self.feature_dim = getattr(self.model, "feature_dim", 512)

    def forward(self, x, return_aux: bool = False):
        if return_aux and hasattr(self.model, '_forward_impl'):
            return self.model(x, return_aux=True)
        return self.model(x)


    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features") and callable(getattr(self.model, "forward_features")):
            return self.model.forward_features(x)
        raise RuntimeError("Internal model does not implement forward_features().")

    @torch.no_grad()
    def get_stream_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.model, "get_stream_features") and callable(getattr(self.model, "get_stream_features")):
            return self.model.get_stream_features(x)
        feat = self.forward_features(x)
        return feat, feat

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_type": self.model_type,
            "inner_model": self.model.__class__.__name__,
            "feature_dim": self.feature_dim,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


def create_classifier(model_name: str,
                      num_classes: int,
                      pretrained: bool = True,
                      ablation: Optional[AblationConfig] = None,
                      **kwargs):
    """
    Factory for all model variants used in the paper.

    Supported model_name values:
      - convnext          : ConvNeXt-only baseline
      - mamba             : Mamba-only baseline
      - dual / dual_naive : ConvNeXt+Mamba naive concat (SAF off, RGBF off)
      - dual_saf          : ConvNeXt+Mamba+SAF (RGBF off)
      - dual_rgbf         : ConvNeXt+Mamba+RGBF (SAF off)
      - dual_full / dual  : ConvNeXt+Mamba+SAF+RGBF (full framework)
      - mobilevit         : MobileViT baseline (from timm)
      - davit             : DaViT baseline (from timm)
    """
    model_name = str(model_name).lower().strip()

    _ = kwargs.pop("fusion_type", None)
    _ = kwargs.pop("fusion_strategy", None)
    _ = kwargs.pop("use_bpaco", None)
    _ = kwargs.pop("bpaco_enabled", None)

    dropout = kwargs.pop("dropout", 0.2)
    fusion_dim = kwargs.pop("fusion_dim", 512)
    mamba_depth = kwargs.pop("mamba_depth", 4)
    mamba_embed_dim = kwargs.pop("mamba_embed_dim", 512)
    mamba_patch_size = kwargs.pop("mamba_patch_size", 16)
    mamba_img_size = kwargs.pop("mamba_img_size", 224)
    convnext_name = kwargs.pop("convnext_name", "convnext_base")
    in_channels = kwargs.pop("in_channels", 3)
    convnext_pretrained = kwargs.pop("convnext_pretrained", pretrained)

    # --- dual variants via ablation config ---
    def _ab(use_saf: bool, use_rgbf_flag: bool) -> AblationConfig:
        return AblationConfig(
            use_convnext=True,
            use_mamba=True,
            use_saf=use_saf,
            saf_prior="edge" if use_saf else "none",
            use_rgbf=use_rgbf_flag,
        ) if ablation is None else ablation

    if model_name in ("dual_full", "dual"):
        if ablation is None:
            ablation = _ab(use_saf=True, use_rgbf_flag=True)
        model = _make_dual(num_classes, pretrained, ablation, convnext_name,
                           convnext_pretrained, in_channels, mamba_img_size,
                           mamba_patch_size, mamba_embed_dim, mamba_depth,
                           fusion_dim, dropout)
        return model

    if model_name == "dual_saf":
        if ablation is None:
            ablation = _ab(use_saf=True, use_rgbf_flag=False)
        model = _make_dual(num_classes, pretrained, ablation, convnext_name,
                           convnext_pretrained, in_channels, mamba_img_size,
                           mamba_patch_size, mamba_embed_dim, mamba_depth,
                           fusion_dim, dropout)
        return model

    if model_name == "dual_rgbf":
        if ablation is None:
            ablation = _ab(use_saf=False, use_rgbf_flag=True)
        model = _make_dual(num_classes, pretrained, ablation, convnext_name,
                           convnext_pretrained, in_channels, mamba_img_size,
                           mamba_patch_size, mamba_embed_dim, mamba_depth,
                           fusion_dim, dropout)
        return model

    if model_name == "dual_naive":
        if ablation is None:
            ablation = _ab(use_saf=False, use_rgbf_flag=False)
        model = _make_dual(num_classes, pretrained, ablation, convnext_name,
                           convnext_pretrained, in_channels, mamba_img_size,
                           mamba_patch_size, mamba_embed_dim, mamba_depth,
                           fusion_dim, dropout)
        return model

    # --- single-stream baselines ---
    if model_name == "convnext":
        model = MedicalImageClassifier(
            model_name="convnext", num_classes=num_classes, pretrained=pretrained,
            in_channels=in_channels, convnext_name=convnext_name, dropout=dropout,
        )
        return model

    if model_name == "mamba":
        model = MedicalImageClassifier(
            model_name="mamba", num_classes=num_classes, pretrained=pretrained,
            in_channels=in_channels, mamba_img_size=mamba_img_size,
            mamba_patch_size=mamba_patch_size, mamba_embed_dim=mamba_embed_dim,
            mamba_depth=mamba_depth, dropout=dropout,
        )
        return model

    # --- external baselines ---
    if model_name in ("mobilevit", "davit"):
        return _create_timm_baseline(model_name, num_classes, pretrained, dropout, **kwargs)

    raise ValueError(f"Unknown model_name: {model_name}. "
                     f"Use convnext/mamba/dual_naive/dual_saf/dual_rgbf/dual_full/mobilevit/davit.")


def _make_dual(num_classes, pretrained, ablation, convnext_name,
               convnext_pretrained, in_channels, mamba_img_size,
               mamba_patch_size, mamba_embed_dim, mamba_depth,
               fusion_dim, dropout):
    return MedicalImageClassifier(
        model_name="dual", num_classes=num_classes, pretrained=pretrained,
        ablation=ablation, convnext_name=convnext_name,
        convnext_pretrained=convnext_pretrained, in_channels=in_channels,
        mamba_img_size=mamba_img_size, mamba_patch_size=mamba_patch_size,
        mamba_embed_dim=mamba_embed_dim, mamba_depth=mamba_depth,
        fusion_dim=fusion_dim, dropout=dropout,
    )


def _create_timm_baseline(model_name: str, num_classes: int, pretrained: bool,
                          dropout: float, **kwargs):
    """Create MobileViT or DaViT baseline via timm."""
    import torch.nn as nn
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for MobileViT/DaViT baselines. "
                          "Install with: pip install timm")

    # Map model aliases to timm model names
    timm_name_map = {
        "mobilevit": ["mobilevit_xxs", "mobilevit_xs", "mobilevit_s"],
        "davit": ["davit_tiny", "davit_small", "davit_base"],
    }

    candidates = timm_name_map.get(model_name, [model_name])
    timm_model_name = None
    for name in candidates:
        try:
            # Check if model is available in timm
            _ = timm.create_model(name, pretrained=False, num_classes=num_classes)
            timm_model_name = name
            break
        except Exception:
            continue

    if timm_model_name is None:
        raise ValueError(
            f"No {model_name} variant found in timm. "
            f"Tried: {candidates}. Please check your timm version."
        )

    backbone = timm.create_model(
        timm_model_name, pretrained=pretrained, num_classes=num_classes,
        drop_rate=dropout,
    )

    class TimmWrapper(nn.Module):
        def __init__(self, backbone, model_type, feat_dim=None):
            super().__init__()
            self.backbone = backbone
            self.model_type = model_type
            if feat_dim is not None:
                self.feature_dim = feat_dim
            elif hasattr(backbone, 'num_features'):
                self.feature_dim = backbone.num_features
            else:
                self.feature_dim = 512

        @torch.no_grad()
        def forward_features(self, x):
            if hasattr(self.backbone, 'forward_features'):
                return self.backbone.forward_features(x)
            if hasattr(self.backbone, 'forward_head'):
                return self.backbone.forward_head(
                    self.backbone.forward_features(x), pre_logits=True
                )
            return self.backbone(x)

        def forward(self, x):
            return self.backbone(x)

    wrapper = TimmWrapper(backbone, model_type=model_name)

    class OuterWrapper(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.model = wrapped
            self.model_name = model_name
            self.feature_dim = wrapped.feature_dim
            self.model_type = model_name

        def forward(self, x):
            return self.model(x)

        def forward_features(self, x):
            return self.model.forward_features(x)

    return OuterWrapper(wrapper)


# =========================================================
# 8) Sanity tests
# =========================================================
def _run_one(model: nn.Module, x: torch.Tensor, name: str, return_aux: bool = False):
    model.eval()
    with torch.no_grad():
        if return_aux and hasattr(model, "model") and hasattr(model.model, "_forward_impl"):
            out = model(x, return_aux=True)
        else:
            out = model(x)
    return out


def test_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Torch] {torch.__version__}")
    B, C, H, W = 2, 3, 224, 224
    dummy_input = torch.randn(B, C, H, W).to(device)

    variants = [
        ("convnext", False),
        ("mamba", False),
        ("dual_naive", False),
        ("dual_saf", False),
        ("dual_rgbf", True),
        ("dual_full", True),
        ("mobilevit", False),
        ("davit", False),
    ]

    n_pass, n_skip, n_fail = 0, 0, 0

    print(f"\n{'='*70}")
    print(f"{'Model':<16} {'Shape':<16} {'Aux':<16} {'Status':<12}")
    print(f"{'-'*70}")

    for name, check_aux in variants:
        try:
            model = create_classifier(name, num_classes=5, pretrained=False).to(device)
            out = _run_one(model, dummy_input, name, return_aux=check_aux)

            if check_aux:
                # Should return (logits, w_c, w_m)
                if isinstance(out, tuple) and len(out) == 3:
                    logits, w_c, w_m = out
                    shape_ok = tuple(logits.shape) == (B, 5)
                    alpha_sum_ok = torch.allclose(w_c + w_m, torch.ones_like(w_c), atol=1e-5)
                    if shape_ok and alpha_sum_ok:
                        print(f"{name:<16} {str(tuple(logits.shape)):<16} {'w_c+w_m≈1 ✓':<16} {'PASS':<12}")
                        n_pass += 1
                    else:
                        print(f"{name:<16} {str(tuple(logits.shape)):<16} {'sum='+str((w_c+w_m)[:2].tolist()):<16} {'FAIL':<12}")
                        n_fail += 1
                else:
                    logits = out[0] if isinstance(out, tuple) else out
                    print(f"{name:<16} {str(tuple(logits.shape)):<16} {'unexpected aux':<16} {'FAIL':<12}")
                    n_fail += 1
            else:
                shape_ok = tuple(out.shape) == (B, 5)
                if shape_ok:
                    print(f"{name:<16} {str(tuple(out.shape)):<16} {'-':<16} {'PASS':<12}")
                    n_pass += 1
                else:
                    print(f"{name:<16} {str(tuple(out.shape)):<16} {'-':<16} {'FAIL':<12}")
                    n_fail += 1

            # Feature dim check
            feat_dim = getattr(model, "feature_dim", None)
            if feat_dim is not None:
                print(f"  {'':>16} feature_dim={feat_dim}")

        except ImportError as e:
            print(f"{name:<16} {'SKIP':<16} {str(e)[:50]:<16} {'SKIP':<12}")
            n_skip += 1
        except Exception as e:
            print(f"{name:<16} {'ERROR':<16} {str(e)[:50]:<16} {'FAIL':<12}")
            n_fail += 1

    print(f"{'='*70}")
    print(f"Results: {n_pass} PASS, {n_skip} SKIP, {n_fail} FAIL")
    print(f"{'='*70}")

    return n_fail == 0


if __name__ == "__main__":
    success = test_all()
    if not success:
        print("\n[WARN] Some model tests failed or were skipped. See details above.")
    else:
        print("\n[OK] All available model tests passed.")

