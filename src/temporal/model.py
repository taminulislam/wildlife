#!/usr/bin/env python3
"""
TTC — Temporal Track Confirmation head (the CVPR contribution).

Replaces the hand-tuned rule
    confirmed = n_frames >= A and span_s >= B and topk_conf >= C
with a learned function of the track's *whole trajectory*.

Design constraints that drove this architecture:
  * ~700 training tracks from 32 videos — the model must be TINY or it memorises.
    2 encoder layers, d_model 64, ~60k parameters.
  * variable-length tracks (1 to hundreds of detections) -> masked attention pooling
    rather than mean/last pooling, so a 3-frame track and a 300-frame track are both
    handled without the padding dominating.
  * the decision depends on temporal PATTERN, not just summary statistics: a real deer
    has smooth motion and sustained confidence; a warm rock flickers and jitters. That
    is exactly what the hand-tuned rule cannot express and what attention can.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AttnPool(nn.Module):
    """Masked attention pooling: learns WHICH timesteps decide the verdict."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x, mask):                      # x (B,T,D), mask (B,T)
        s = self.score(x).squeeze(-1)                # (B,T)
        s = s.masked_fill(mask < 0.5, float("-inf"))
        w = torch.softmax(s, dim=1).unsqueeze(-1)    # (B,T,1)
        return (x * w).sum(1), w.squeeze(-1)


class TemporalTrackNet(nn.Module):
    def __init__(self, n_feat: int = 10, d_model: int = 64, nhead: int = 4,
                 layers: int = 2, dropout: float = 0.2, n_ctx: int = 8):
        super().__init__()
        # cross-track context is fused AFTER pooling: it describes the track's relation
        # to its competitors, not any single timestep.
        self.ctx = nn.Sequential(nn.Linear(n_ctx, d_model // 2), nn.GELU(),
                                 nn.LayerNorm(d_model // 2)) if n_ctx else None
        self.inp = nn.Sequential(nn.Linear(n_feat, d_model), nn.GELU(),
                                 nn.LayerNorm(d_model))
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.pool = AttnPool(d_model)
        fuse = d_model + (d_model // 2 if n_ctx else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fuse, d_model // 2),
                                  nn.GELU(), nn.Linear(d_model // 2, 1))

    def forward(self, x, mask, ctx=None, return_attn: bool = False):
        h = self.pos(self.inp(x))
        h = self.enc(h, src_key_padding_mask=(mask < 0.5))
        pooled, attn = self.pool(h, mask)
        if self.ctx is not None and ctx is not None:
            pooled = torch.cat([pooled, self.ctx(ctx)], dim=-1)
        logit = self.head(pooled).squeeze(-1)
        return (logit, attn) if return_attn else logit


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
