"""SEC284-L: fixed-task visual condition distillation.

The module deliberately has no action, state, LAP, LaWM, Expert, or language
input.  It maps frozen three-view DINO tokens directly to the 284x768 semantic
condition consumed by the released Action Expert's VLM-free ``h_lap`` path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SEC284Config:
    """Immutable SEC284-L architecture contract."""

    num_views: int = 3
    tokens_per_view: int = 256
    vision_dim: int = 768
    model_dim: int = 768
    output_dim: int = 768
    num_queries: int = 284
    num_layers: int = 8
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.num_views != 3:
            raise ValueError("SEC284-L is defined for exactly three views")
        if self.tokens_per_view != 256:
            raise ValueError("SEC284-L is defined for 256 DINO tokens per view")
        if self.vision_dim != 768 or self.model_dim != 768 or self.output_dim != 768:
            raise ValueError("SEC284-L width contract is fixed at 768")
        if self.num_queries != 284:
            raise ValueError("SEC284-L output contract is fixed at 284 tokens")
        if self.num_layers != 8 or self.num_heads != 12 or self.ffn_dim != 3072:
            raise ValueError("SEC284-L capacity contract is 8x768, 12 heads, FFN 3072")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")


class SEC284DecoderBlock(nn.Module):
    """Pre-norm query self-attention, visual cross-attention, and FFN."""

    def __init__(self, config: SEC284Config) -> None:
        super().__init__()
        dim = config.model_dim
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            config.num_heads,
            kdim=dim,
            vdim=dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = self.norm_self(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        q = self.norm_cross(queries)
        queries = queries + self.cross_attn(
            q,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        return queries + self.ffn(self.norm_ffn(queries))


class SEC284L(nn.Module):
    """76,624,896-parameter fixed-task visual condition student."""

    def __init__(self, config: SEC284Config | None = None) -> None:
        super().__init__()
        self.config = config or SEC284Config()
        c = self.config
        self.task_queries = nn.Parameter(torch.empty(c.num_queries, c.model_dim))
        self.view_embeddings = nn.Parameter(torch.empty(c.num_views, c.model_dim))
        self.patch_embeddings = nn.Parameter(torch.empty(c.tokens_per_view, c.model_dim))
        self.input_norm = nn.LayerNorm(c.model_dim)
        self.blocks = nn.ModuleList([SEC284DecoderBlock(c) for _ in range(c.num_layers)])
        self.output_norm = nn.LayerNorm(c.model_dim)
        self.output_proj = nn.Linear(c.model_dim, c.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.task_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.view_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.patch_embeddings, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def checkpoint_config(self) -> dict[str, int | float]:
        return asdict(self.config)

    def _validate_inputs(
        self, visual_tokens: torch.Tensor, view_mask: torch.Tensor | None
    ) -> torch.Tensor:
        c = self.config
        if visual_tokens.ndim != 4:
            raise ValueError(
                f"visual_tokens must be [B,3,256,768], got {tuple(visual_tokens.shape)}"
            )
        if tuple(visual_tokens.shape[1:]) != (c.num_views, c.tokens_per_view, c.vision_dim):
            raise ValueError(
                f"visual_tokens must be [B,{c.num_views},{c.tokens_per_view},{c.vision_dim}], "
                f"got {tuple(visual_tokens.shape)}"
            )
        batch = visual_tokens.shape[0]
        if view_mask is None:
            return torch.ones(batch, c.num_views, device=visual_tokens.device, dtype=torch.bool)
        if tuple(view_mask.shape) != (batch, c.num_views):
            raise ValueError(
                f"view_mask must be [B,{c.num_views}], got {tuple(view_mask.shape)}"
            )
        mask = view_mask.to(device=visual_tokens.device, dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each SEC284-L sample must retain at least one valid view")
        return mask

    def forward(
        self,
        visual_tokens: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the VLM-compatible condition with shape ``[B,284,768]``."""
        c = self.config
        valid_views = self._validate_inputs(visual_tokens, view_mask)
        batch = visual_tokens.shape[0]
        x = visual_tokens
        x = x + self.view_embeddings.view(1, c.num_views, 1, c.model_dim)
        x = x + self.patch_embeddings.view(1, 1, c.tokens_per_view, c.model_dim)
        memory = self.input_norm(x).reshape(batch, c.num_views * c.tokens_per_view, c.model_dim)
        memory_padding_mask = (~valid_views).unsqueeze(-1).expand(
            batch, c.num_views, c.tokens_per_view
        ).reshape(batch, c.num_views * c.tokens_per_view)
        queries = self.task_queries.unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, memory_padding_mask)
        condition = self.output_proj(self.output_norm(queries))
        if tuple(condition.shape[1:]) != (c.num_queries, c.output_dim):
            raise RuntimeError(f"SEC284-L internal output mismatch: {tuple(condition.shape)}")
        if not torch.isfinite(condition).all():
            raise RuntimeError("SEC284-L produced non-finite condition values")
        return condition
