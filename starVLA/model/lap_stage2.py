"""VLM-free LAP8 semantic conditioner used by Stage-2 action training.

LAP8 preserves the Stage-1 LAP6 latent-action path exactly.  Two additional
fusion blocks form a separate Action-Expert branch, so fine-tuning the branch
cannot move the tap point used by LaWM.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from starVLA.model.lap_stage1 import LAP60M, LAPFusionBlock


class LAP8(nn.Module):
    """Frozen LAP6 plus a trainable two-block Expert-conditioning branch."""

    def __init__(
        self,
        lap6: LAP60M,
        *,
        num_extra_layers: int = 2,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        view_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_extra_layers != 2:
            raise ValueError("LAP8 requires exactly two Expert-only fusion blocks")
        if lap6.num_views != 3:
            raise ValueError(f"LAP8 expects a three-view LAP6, got {lap6.num_views} views")
        if not 0.0 <= view_dropout < 1.0:
            raise ValueError("view_dropout must be in [0, 1)")

        self.lap6 = lap6
        self.view_dropout = float(view_dropout)
        for parameter in self.lap6.parameters():
            parameter.requires_grad_(False)
        self.lap6.eval()

        dim = int(lap6.vision_dim)
        self.task_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.latent_to_expert = nn.Linear(int(lap6.latent_dim), dim)
        self.expert_fusion = nn.ModuleList(
            [
                LAPFusionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_extra_layers)
            ]
        )
        self.lap_to_expert = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

        nn.init.normal_(self.task_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.latent_to_expert.weight)
        nn.init.zeros_(self.latent_to_expert.bias)
        # Start as a feature-preserving adapter. LayerNorm remains the same
        # normalization used by the Expert's former VLM projection.
        nn.init.eye_(self.lap_to_expert[0].weight)
        nn.init.zeros_(self.lap_to_expert[0].bias)

    def train(self, mode: bool = True) -> "LAP8":
        super().train(mode)
        # Parent .train() would otherwise enable Stage-1 view dropout and any
        # dropout inside LAP6.  The frozen trunk must be deterministic.
        self.lap6.eval()
        return self

    def _prepare_visual_context(
        self, visual_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if visual_tokens.ndim != 4:
            raise ValueError(
                f"LAP8 visual_tokens must be [B,3,K,D], got {tuple(visual_tokens.shape)}"
            )
        batch, views, tokens_per_view, width = visual_tokens.shape
        if views != self.lap6.num_views or width != self.lap6.vision_dim:
            raise ValueError(
                f"expected [B,{self.lap6.num_views},K,{self.lap6.vision_dim}], "
                f"got {tuple(visual_tokens.shape)}"
            )
        visual = visual_tokens + self.lap6.view_embeddings.detach().view(1, views, 1, width)
        visual_padding_mask = None
        if self.training and self.view_dropout > 0.0:
            dropped = torch.rand(batch, views, device=visual.device) < self.view_dropout
            dropped[:, 0] = False
            visual_padding_mask = dropped.unsqueeze(-1).expand(-1, -1, tokens_per_view)
            visual_padding_mask = visual_padding_mask.reshape(batch, views * tokens_per_view)
        return visual.reshape(batch, views * tokens_per_view, width), visual_padding_mask

    def forward(self, visual_tokens: torch.Tensor, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            stage1 = self.lap6(visual_tokens, state)
        z_lap = stage1["z_lap"]
        scene6 = stage1["scene_tokens"]

        visual, visual_padding_mask = self._prepare_visual_context(visual_tokens)
        queries = (
            scene6
            + self.task_embedding
            + self.latent_to_expert(z_lap.squeeze(1)).unsqueeze(1)
        )
        for block in self.expert_fusion:
            queries = block(queries, visual, visual_padding_mask)
        cond_lap = self.lap_to_expert(queries)
        return {
            "z_lap": z_lap,
            "scene_lap6": scene6,
            "scene_lap8": queries,
            "cond_lap": cond_lap,
        }


class LAP10(nn.Module):
    """LAP8 plus two Expert-interface blocks that reproduce VLM token length.

    The released Expert consumes the complete projected VLM sequence.  For the
    fixed Task-14 prompt and three 256px views that sequence has 284 tokens.
    LAP10 expands LAP8's eight semantic tokens into the same ``[B,284,768]``
    interface without allocating a VLM at inference time.
    """

    def __init__(
        self,
        lap8: LAP8,
        *,
        output_tokens: int = 284,
        num_layers: int = 2,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("LAP10 requires exactly two interface-alignment blocks")
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        self.lap8 = lap8
        self.output_tokens = int(output_tokens)
        dim = int(lap8.lap6.vision_dim)
        self.output_queries = nn.Parameter(torch.empty(output_tokens, dim))
        self.interface_fusion = nn.ModuleList(
            [
                LAPFusionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.interface_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.output_queries, mean=0.0, std=0.02)

    def train(self, mode: bool = True) -> "LAP10":
        super().train(mode)
        # LAP8 in turn keeps its frozen LAP6 trunk deterministic.
        self.lap8.train(mode)
        return self

    def forward(self, visual_tokens: torch.Tensor, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        lap8_out = self.lap8(visual_tokens, state)
        memory = lap8_out["cond_lap"]
        queries = self.output_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        for block in self.interface_fusion:
            queries = block(queries, memory)
        result = dict(lap8_out)
        result["cond_lap8"] = memory
        result["cond_lap10"] = self.interface_norm(queries)
        return result


class LAP10V2(nn.Module):
    """Unified 284-token Expert branch built directly on the frozen LAP6 trunk.

    LAP6 continues to produce the stable ``z_lap`` used by LaWM.  The four
    post-LAP6 blocks all operate on the final 284 Expert queries and attend to
    the complete three-view DINO memory plus the eight LAP6 scene tokens.
    """

    def __init__(
        self,
        lap8: LAP8,
        *,
        output_tokens: int = 284,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lap8 = lap8
        self.output_tokens = int(output_tokens)
        dim = int(lap8.lap6.vision_dim)
        self.output_queries = nn.Parameter(torch.empty(output_tokens, dim))
        self.interface_fusion = nn.ModuleList(
            [LAPFusionBlock(dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in range(2)]
        )
        self.interface_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.output_queries, mean=0.0, std=0.02)
        # This projection belongs to the former 8-token output head and is not
        # used in the unified branch.  Keep it frozen so DDP has no unused
        # trainable parameter.
        for parameter in self.lap8.lap_to_expert.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "LAP10V2":
        super().train(mode)
        self.lap8.lap6.eval()
        return self

    def load_from_lap10_state(self, state: dict[str, torch.Tensor]) -> None:
        """Initialize shared layers from the trained, bottlenecked LAP10."""
        self.output_queries.data.copy_(state["output_queries"])
        self.interface_norm.load_state_dict({
            key.removeprefix("interface_norm."): value
            for key, value in state.items() if key.startswith("interface_norm.")
        })
        self.interface_fusion.load_state_dict({
            key.removeprefix("interface_fusion."): value
            for key, value in state.items() if key.startswith("interface_fusion.")
        })

    def forward(self, visual_tokens: torch.Tensor, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Preserve the already-validated LAP6 -> z_lap -> LaWM interface.
        with torch.no_grad():
            stage1 = self.lap8.lap6(visual_tokens, state)
        z_lap = stage1["z_lap"]
        scene6 = stage1["scene_tokens"]
        visual, visual_padding_mask = self.lap8._prepare_visual_context(visual_tokens)
        memory = torch.cat([scene6, visual], dim=1)
        if visual_padding_mask is not None:
            prefix = torch.zeros(
                visual_padding_mask.shape[0], scene6.shape[1], device=visual_padding_mask.device, dtype=torch.bool
            )
            memory_padding_mask = torch.cat([prefix, visual_padding_mask], dim=1)
        else:
            memory_padding_mask = None

        queries = (
            self.output_queries.unsqueeze(0).expand(visual.shape[0], -1, -1)
            + self.lap8.task_embedding
            + self.lap8.latent_to_expert(z_lap.squeeze(1)).unsqueeze(1)
        )
        # Blocks 7-8: the trained LAP8 blocks, now operating directly on 284 queries.
        for block in self.lap8.expert_fusion:
            queries = block(queries, memory, memory_padding_mask)
        scene_lap8 = queries
        # Blocks 9-10: the former LAP10 blocks, with the same full memory.
        for block in self.interface_fusion:
            queries = block(queries, memory, memory_padding_mask)
        return {
            "z_lap": z_lap,
            "scene_lap6": scene6,
            "scene_lap8": scene_lap8,
            "cond_lap10": self.interface_norm(queries),
        }


class LAPRoleFusionBlock(nn.Module):
    """Decoder block that re-injects a distinct role embedding at each layer."""

    def __init__(self, dim: int = 768, num_heads: int = 12, ffn_dim: int = 3072,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, kdim=dim, vdim=dim, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, queries: torch.Tensor, roles: torch.Tensor,
                memory: torch.Tensor, memory_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q_norm = self.norm_self(queries)
        q_role = q_norm + roles
        queries = queries + self.self_attn(q_role, q_role, q_norm, need_weights=False)[0]
        q_role = self.norm_cross(queries) + roles
        queries = queries + self.cross_attn(
            q_role, memory, memory, key_padding_mask=memory_padding_mask, need_weights=False
        )[0]
        return queries + self.ffn(self.norm_ffn(queries))


class LAP10V3(nn.Module):
    """From-scratch four-layer (LAP7-LAP10) 284-token conditioner.

    Only the supplied LAP6 trunk is loaded/frozen.  The post-LAP6 branch uses a
    per-position teacher mean as a fixed output template and learns a visual
    residual, while role embeddings are re-injected in every decoder block.
    """

    def __init__(self, lap6: LAP60M, position_mean: torch.Tensor, *, output_tokens: int = 284,
                 num_layers: int = 4, num_heads: int = 12, ffn_dim: int = 3072,
                 dropout: float = 0.0, view_dropout: float = 0.0) -> None:
        super().__init__()
        if num_layers != 4:
            raise ValueError("LAP10V3 requires four post-LAP6 blocks (LAP7-LAP10)")
        if lap6.num_views != 3:
            raise ValueError("LAP10V3 expects a three-view LAP6 trunk")
        if tuple(position_mean.shape) != (output_tokens, lap6.vision_dim):
            raise ValueError(f"position_mean must be [{output_tokens},{lap6.vision_dim}]")
        self.lap6 = lap6
        self.output_tokens = int(output_tokens)
        self.view_dropout = float(view_dropout)
        dim = int(lap6.vision_dim)
        for parameter in self.lap6.parameters():
            parameter.requires_grad_(False)
        self.lap6.eval()

        self.content_queries = nn.Parameter(torch.randn(output_tokens, dim) * 0.02)
        self.role_embeddings = nn.Parameter(torch.randn(output_tokens, dim) * 0.02)
        self.view_embeddings = nn.Parameter(torch.randn(lap6.num_views, dim) * 0.02)
        self.latent_to_memory = nn.Linear(int(lap6.latent_dim), dim)
        self.state_to_memory = nn.Sequential(nn.Linear(int(lap6.state_dim), dim), nn.LayerNorm(dim), nn.GELU())
        self.blocks = nn.ModuleList([
            LAPRoleFusionBlock(dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.residual_norm = nn.LayerNorm(dim)
        self.residual_head = nn.Linear(dim, dim)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_head.bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("teacher_position_mean", position_mean.detach().float().unsqueeze(0))

    def train(self, mode: bool = True) -> "LAP10V3":
        super().train(mode)
        self.lap6.eval()
        return self

    def _visual_memory(self, visual_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if visual_tokens.ndim != 4 or visual_tokens.shape[1] != self.lap6.num_views:
            raise ValueError(f"expected [B,3,K,{self.lap6.vision_dim}], got {tuple(visual_tokens.shape)}")
        b, views, tokens, width = visual_tokens.shape
        visual = visual_tokens + self.view_embeddings.view(1, views, 1, width)
        padding = None
        if self.training and self.view_dropout > 0.0:
            dropped = torch.rand(b, views, device=visual.device) < self.view_dropout
            dropped[:, 0] = False
            padding = dropped.unsqueeze(-1).expand(-1, -1, tokens).reshape(b, views * tokens)
        return visual.reshape(b, views * tokens, width), padding

    def forward(self, visual_tokens: torch.Tensor, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            stage1 = self.lap6(visual_tokens, state)
        z_lap, scene6 = stage1["z_lap"], stage1["scene_tokens"]
        visual, visual_padding = self._visual_memory(visual_tokens)
        latent = self.latent_to_memory(z_lap.squeeze(1)).unsqueeze(1)
        state_token = self.state_to_memory(state).unsqueeze(1)
        memory = torch.cat([visual, scene6, latent, state_token], dim=1)
        memory_padding = None
        if visual_padding is not None:
            prefix = torch.zeros(
                visual_padding.shape[0], scene6.shape[1] + 2,
                device=visual_padding.device, dtype=torch.bool
            )
            memory_padding = torch.cat([visual_padding, prefix], dim=1)
        queries = self.content_queries.unsqueeze(0).expand(visual.shape[0], -1, -1)
        roles = self.role_embeddings.unsqueeze(0).expand_as(queries)
        for block in self.blocks:
            queries = block(queries, roles, memory, memory_padding)
        residual = self.residual_head(self.residual_norm(queries)) * self.residual_scale
        cond_lap = self.teacher_position_mean + residual
        return {
            "z_lap": z_lap,
            "scene_lap6": scene6,
            "dynamic_residual": residual,
            "cond_lap": cond_lap,
        }
