"""
Category-conditioned name generation model.

- Trunk: one list of blocks, applied as x = x + block(x) (identity shortcut only).
- Shortcuts are never weighted: residual only where input and output dimensions match.
- Configurable depths and dimensions via ModelConfig.
- Inherits GenerationMixin for generate_token / generate_name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from part6.generation import GenerationMixin


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for CategoryConditionedNameModel."""

    vocab_size: int
    cat_vocab_size: int
    block_size: int
    n_embd: int = 64
    n_hidden: int = 512
    num_heads: int = 8
    num_attention_blocks: int = 5
    num_mlp_layers: int = 1
    cat_emb_dim: int = 32
    dropout: float = 0.1
    last_layer_scale: float = 0.1
    category_dropout: float = 0.0  # probability of replacing category with unknown (for generalization)
    unknown_category_idx: int = 0  # index used when category is dropped (e.g. cat_stoi["unknown"])

    @property
    def head_size(self) -> int:
        return self.n_hidden

    @property
    def flatten_size(self) -> int:
        return self.block_size * self.n_hidden


# ---------------------------------------------------------------------------
# Trunk blocks: each takes (x, context) and returns same shape as x
# ---------------------------------------------------------------------------


class AdaptiveLayerNorm(nn.Module):
    """
    AdaLN: normalize x then scale and shift by category embedding.
    out = scale(cat_emb) * LayerNorm(x) + shift(cat_emb).
    Same API as other trunk components: forward(x, cat_emb) -> same shape as x.
    """

    def __init__(self, cat_emb_dim: int, n_embd: int):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.cond_proj = nn.Linear(cat_emb_dim, 2 * n_embd)

    def forward(self, x: torch.Tensor, cat_emb: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln(x)
        scale, shift = self.cond_proj(cat_emb).chunk(2, dim=-1)
        return scale.unsqueeze(1) * x_norm + shift.unsqueeze(1)


class AdaLNBlockWithContext(nn.Module):
    """Wraps AdaptiveLayerNorm so that x = x + block(x, context) yields x = adaln(x)."""

    def __init__(self, adaln: AdaptiveLayerNorm):
        super().__init__()
        self.adaln = adaln

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        cat_emb = context["cat_emb"] if context else None
        if cat_emb is None:
            raise ValueError("AdaLN block requires context['cat_emb']")
        out = self.adaln(x, cat_emb)
        return out - x


class CategoryCrossAttention(nn.Module):
    """Category vector queries the sequence (q from cat, k,v from seq). Pre-norm, residual."""

    def __init__(
        self,
        cat_emb_dim: int,
        n_embd: int,
        head_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_x = nn.LayerNorm(n_embd)
        self.ln_cat = nn.LayerNorm(cat_emb_dim)
        self.cat_query = nn.Linear(cat_emb_dim, head_size, bias=False)
        self.seq_key = nn.Linear(n_embd, head_size, bias=False)
        self.seq_value = nn.Linear(n_embd, head_size, bias=False)
        self.proj = nn.Linear(head_size, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cat_emb: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_norm = self.ln_x(x)
        cat_norm = self.ln_cat(cat_emb)
        q = self.cat_query(cat_norm).unsqueeze(1)  # (B, 1, head_size)
        k = self.seq_key(x_norm)    # (B, T, head_size)
        v = self.seq_value(x_norm)  # (B, T, head_size)
        wei = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        wei = F.softmax(wei, dim=-1)
        out = wei @ v  # (B, 1, head_size)
        out = out.expand(B, T, -1)
        return x + self.dropout(self.proj(out))


class SequenceQueriesCategoryAttention(nn.Module):
    """Sequence queries the category (q from seq, k,v from cat). Pre-norm, residual. Increases interplay."""

    def __init__(
        self,
        cat_emb_dim: int,
        n_embd: int,
        head_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_x = nn.LayerNorm(n_embd)
        self.ln_cat = nn.LayerNorm(cat_emb_dim)
        self.seq_query = nn.Linear(n_embd, head_size, bias=False)
        self.cat_key = nn.Linear(cat_emb_dim, head_size, bias=False)
        self.cat_value = nn.Linear(cat_emb_dim, head_size, bias=False)
        self.proj = nn.Linear(head_size, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cat_emb: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_norm = self.ln_x(x)
        cat_norm = self.ln_cat(cat_emb)  # (B, cat_emb_dim)
        q = self.seq_query(x_norm)  # (B, T, head_size)
        k = self.cat_key(cat_norm).unsqueeze(1)    # (B, 1, head_size)
        v = self.cat_value(cat_norm).unsqueeze(1)  # (B, 1, head_size)
        wei = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)  # (B, T, 1)
        wei = F.softmax(wei, dim=-1)
        out = wei @ v  # (B, T, head_size)
        return x + self.dropout(self.proj(out))


class TrunkBlockWithContext(nn.Module):
    """Wraps a block that needs (x, cat_emb) so it has signature (x, context)."""

    def __init__(self, cross_attn: CategoryCrossAttention | SequenceQueriesCategoryAttention):
        super().__init__()
        self.cross_attn = cross_attn

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        cat_emb = context["cat_emb"] if context else None
        if cat_emb is None:
            raise ValueError("Category cross-attention block requires context['cat_emb']")
        return self.cross_attn(x, cat_emb)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention. Pre-norm, residual."""

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_embd % num_heads == 0
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        self.ln = nn.LayerNorm(n_embd)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        del context  # unused
        B, T, C = x.shape
        x_norm = self.ln(x)
        k = self.key(x_norm).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = self.query(x_norm).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x_norm).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        return x + self.dropout(self.proj(out))


# ---------------------------------------------------------------------------
# MLP head: residual only when in_dim == out_dim (identity shortcut)
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """
    Pre-norm MLP: ln(x) -> linear -> act -> dropout, then residual if in_dim == out_dim.
    When use_ln=False (e.g. final logits layer), no LayerNorm is applied.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        activation: str = "tanh",
        use_ln: bool = True,
    ):
        super().__init__()
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(in_dim) if use_ln else nn.Identity()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act: Callable[[torch.Tensor], torch.Tensor] = (
            torch.tanh if activation == "tanh" else (lambda x: x)
        )
        self._residual = in_dim == out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.dropout(self.linear(self.ln(x))))
        if self._residual:
            h = x + h
        return h


def _mlp_blocks_spec(config: ModelConfig) -> list[tuple[int, int, str, float, bool]]:
    """(in_dim, out_dim, activation, dropout, use_ln) per block. Last block: no ln, no dropout."""
    flat = config.flatten_size
    hidden = config.n_hidden
    vocab = config.vocab_size
    n = config.num_mlp_layers
    drop = config.dropout
    spec: list[tuple[int, int, str, float, bool]] = []
    in_d = flat
    for _ in range(n):
        spec.append((in_d, hidden, "tanh", drop, True))
        in_d = hidden
    spec.append((in_d, vocab, "identity", 0.0, False))  # final logits: no LayerNorm
    return spec


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class CategoryConditionedNameModel(GenerationMixin, nn.Module):
    """
    Trunk: embed -> project to n_hidden -> for block in blocks: x = x + block(x, context).
    Head: flatten -> MLP blocks (residual only when dims match).
    Shortcuts are identity only; no learned shortcut projections.
    Generation: generate_token, generate_name (from GenerationMixin), plus generate() convenience.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__()
        self.config = config or ModelConfig(**kwargs)
        c = self.config

        self.token_embedding = nn.Embedding(c.vocab_size, c.n_embd)
        self.cat_embedding = nn.Embedding(c.cat_vocab_size, c.cat_emb_dim)

        # Single projection (no residual: dimension change n_embd -> n_hidden)
        self.project = nn.Linear(c.n_embd, c.n_hidden, bias=False)

        # Trunk: AdaLN (category modulates sequence first), then cat↔seq cross-attn, then self-attention
        adaln = AdaptiveLayerNorm(c.cat_emb_dim, c.n_hidden)
        cat_cross = CategoryCrossAttention(
            c.cat_emb_dim, c.n_hidden, c.head_size, c.dropout
        )
        seq_queries_cat = SequenceQueriesCategoryAttention(
            c.cat_emb_dim, c.n_hidden, c.head_size, c.dropout
        )
        self.blocks = nn.ModuleList([
            AdaLNBlockWithContext(adaln),
            TrunkBlockWithContext(cat_cross),
            TrunkBlockWithContext(seq_queries_cat),
            *[
                MultiHeadSelfAttention(c.n_hidden, c.num_heads, c.block_size, c.dropout)
                for _ in range(c.num_attention_blocks)
            ],
        ])

        self.mlp_blocks = nn.ModuleList([
            MLPBlock(in_d, out_d, drop, act, use_ln=use_ln)
            for in_d, out_d, act, drop, use_ln in _mlp_blocks_spec(c)
        ])

        self._init_weights(c.last_layer_scale)

    def _init_weights(self, last_layer_scale: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        last_block = self.mlp_blocks[-1]
        if last_block.use_ln:
            last_block.ln.weight.data.mul_(last_layer_scale)
            last_block.ln.bias.data.mul_(last_layer_scale)
        else:
            last_block.linear.weight.data.mul_(last_layer_scale)

    @property
    def block_size(self) -> int:
        return self.config.block_size

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def forward(
        self,
        idx: torch.Tensor,
        cat_idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        c = self.config

        # Category dropout: during training, replace category with unknown with probability category_dropout
        if self.training and targets is not None and c.category_dropout > 0:
            drop = torch.rand(B, device=cat_idx.device) < c.category_dropout
            cat_idx = torch.where(
                drop,
                torch.full((B,), c.unknown_category_idx, device=cat_idx.device, dtype=cat_idx.dtype),
                cat_idx,
            )

        x = self.project(self.token_embedding(idx))
        context = {"cat_emb": self.cat_embedding(cat_idx)}

        for block in self.blocks:
            x = x + block(x, context=context)

        x = x.reshape(B, -1)
        for block in self.mlp_blocks:
            x = block(x)

        logits = x
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss
