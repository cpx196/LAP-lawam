"""The single-task 60M Latent Action Predictor used by Stage 1.

This module deliberately contains no language encoder and no task-id input.  The
task is represented by the weights and the learned scene queries.  It consumes
the frozen DINO/V-JEPA feature map at the current frame and the current 16-D
dual-arm EEF state and predicts the 32-D latent action used by LaWM.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class LAPFusionBlock(nn.Module):
    """One scene-query self-attention + visual cross-attention block."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            kdim=dim,
            vdim=dim,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        visual: torch.Tensor,
        visual_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm_self(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]

        q = self.norm_cross(queries)
        queries = queries + self.cross_attn(
            q,
            visual,
            visual,
            key_padding_mask=visual_padding_mask,
            need_weights=False,
        )[0]

        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class LAP60M(nn.Module):
    """Approximately 60M parameter single-task LAP.

    Args:
        vision_dim: Frozen visual-token width.  RoboTwin's DINO ViT-B/16
            checkpoint produces 768-wide, 16x16=256-token features.
        state_dim: Current dual-arm EEF state width (3+4+1 per arm = 16).
        latent_dim: IDM/LaWM latent action width (32 in the released LAM).
    """

    def __init__(
        self,
        vision_dim: int = 768,
        state_dim: int = 16,
        latent_dim: int = 32,
        num_scene_queries: int = 8,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        num_views: int = 3,
        view_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if vision_dim % num_heads != 0:
            raise ValueError(f"vision_dim={vision_dim} must be divisible by num_heads={num_heads}")
        self.vision_dim = int(vision_dim)
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.num_scene_queries = int(num_scene_queries)
        self.num_views = int(num_views)
        self.view_dropout = float(view_dropout)
        if self.num_views < 1:
            raise ValueError("num_views must be >= 1")
        if not 0.0 <= self.view_dropout < 1.0:
            raise ValueError("view_dropout must be in [0, 1)")

        # A state token is kept at full width so it can participate in every
        # fusion layer without a separate low-dimensional bottleneck.
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, vision_dim),
            nn.LayerNorm(vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, vision_dim),
        )
        self.scene_queries = nn.Parameter(torch.randn(num_scene_queries, vision_dim) * 0.02)
        self.view_embeddings = nn.Parameter(torch.randn(num_views, vision_dim) * 0.02)
        self.fusion = nn.ModuleList(
            [
                LAPFusionBlock(
                    dim=vision_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.pool_query = nn.Parameter(torch.randn(1, vision_dim) * 0.02)
        self.pool_norm = nn.LayerNorm(vision_dim)
        self.pool_attn = nn.MultiheadAttention(
            vision_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.latent_norm = nn.LayerNorm(vision_dim)
        self.latent_head = nn.Linear(vision_dim, latent_dim)

    def forward(self, visual_tokens: torch.Tensor, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict the latent action from the current observation.

        Args:
            visual_tokens: ``[B, 256, 768]`` frozen DINO features.
            state: ``[B, 16]`` current absolute dual-arm EEF state.
        Returns:
            ``z_lap`` has shape ``[B, 1, 32]`` and ``scene_tokens`` has shape
            ``[B, 8, 768]``.
        """
        if visual_tokens.ndim not in (3, 4):
            raise ValueError(
                f"visual_tokens must be [B,K,D] or [B,V,K,D], got {tuple(visual_tokens.shape)}"
            )
        if state.ndim != 2:
            raise ValueError(f"state must be [B,D], got {tuple(state.shape)}")
        if visual_tokens.shape[-1] != self.vision_dim:
            raise ValueError(
                f"visual feature width mismatch: expected {self.vision_dim}, "
                f"got {visual_tokens.shape[-1]}"
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"EEF state width mismatch: expected {self.state_dim}, got {state.shape[-1]}"
            )

        batch = visual_tokens.shape[0]
        visual_padding_mask = None
        if visual_tokens.ndim == 3:
            visual_tokens = visual_tokens + self.view_embeddings[0].view(1, 1, -1)
        else:
            views, tokens_per_view = visual_tokens.shape[1:3]
            if views != self.num_views:
                raise ValueError(f"expected {self.num_views} views, got {views}")
            visual_tokens = visual_tokens + self.view_embeddings.view(1, views, 1, -1)
            if self.training and self.view_dropout > 0.0 and views > 1:
                # The main view (index 0) is always present.  Each wrist view
                # is independently dropped to make deployment robust to
                # occlusion or a missing camera.
                dropped = torch.rand(batch, views, device=visual_tokens.device) < self.view_dropout
                dropped[:, 0] = False
                visual_padding_mask = dropped.unsqueeze(-1).expand(-1, -1, tokens_per_view)
                visual_padding_mask = visual_padding_mask.reshape(batch, views * tokens_per_view)
            visual_tokens = visual_tokens.reshape(batch, views * tokens_per_view, self.vision_dim)
        scene = self.scene_queries.unsqueeze(0).expand(batch, -1, -1)
        state_token = self.state_encoder(state).unsqueeze(1)
        queries = torch.cat([scene, state_token], dim=1)
        for block in self.fusion:
            queries = block(queries, visual_tokens, visual_padding_mask)

        # The state token is used for prediction but is not exported as a
        # scene token; Stage 2 receives exactly the eight learned scene tokens.
        scene_tokens = queries[:, : self.num_scene_queries]
        pool_query = self.pool_query.unsqueeze(0).expand(batch, -1, -1)
        pooled, _ = self.pool_attn(
            self.pool_norm(pool_query),
            queries,
            queries,
            need_weights=False,
        )
        latent = self.latent_head(self.latent_norm(pooled))
        return {
            "z_lap": latent,
            "scene_tokens": scene_tokens,
            "pooled": pooled,
        }


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Return a stable parameter count for logging and config validation."""
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )
