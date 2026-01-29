"""
Category-conditioned name generation model.

- Trunk: one list of blocks, applied as x = x + block(x) (identity shortcut only).
- Shortcuts are never weighted: residual only where input and output dimensions match.
- Configurable depths and dimensions via ModelConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    @property
    def head_size(self) -> int:
        return self.n_hidden

    @property
    def flatten_size(self) -> int:
        return self.block_size * self.n_hidden


# ---------------------------------------------------------------------------
# Trunk blocks: each takes (x, context) and returns same shape as x
# ---------------------------------------------------------------------------


class CategoryCrossAttention(nn.Module):
    """Category vector queries the character sequence. Same in/out shape as x."""

    def __init__(
        self,
        cat_emb_dim: int,
        n_embd: int,
        head_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cat_query = nn.Linear(cat_emb_dim, head_size, bias=False)
        self.seq_key = nn.Linear(n_embd, head_size, bias=False)
        self.seq_value = nn.Linear(n_embd, head_size, bias=False)
        self.proj = nn.Linear(head_size, n_embd, bias=False)
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cat_emb: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.cat_query(cat_emb).unsqueeze(1)  # (B, 1, head_size)
        k = self.seq_key(x)    # (B, T, head_size)
        v = self.seq_value(x)  # (B, T, head_size)
        wei = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        wei = F.softmax(wei, dim=-1)
        out = wei @ v  # (B, 1, head_size)
        out = out.expand(B, T, -1)
        return self.dropout(self.ln(self.proj(out)))


class TrunkBlockWithContext(nn.Module):
    """Wraps a block that needs extra args (e.g. cat_emb) so it has signature (x, context)."""

    def __init__(self, cross_attn: CategoryCrossAttention):
        super().__init__()
        self.cross_attn = cross_attn

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        cat_emb = context["cat_emb"] if context else None
        if cat_emb is None:
            raise ValueError("Category cross-attention block requires context['cat_emb']")
        return self.cross_attn(x, cat_emb)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention; input and output (B, T, n_embd)."""

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
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        del context  # unused
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.ln(self.proj(out)))


# ---------------------------------------------------------------------------
# MLP head: residual only when in_dim == out_dim (identity shortcut)
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """
    One block: out = activation(LayerNorm(Dropout(linear(x)))) with optional identity shortcut.
    Shortcut is used only when in_dim == out_dim (no extra weights on the shortcut).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        activation: str = "tanh",
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act: Callable[[torch.Tensor], torch.Tensor] = (
            torch.tanh if activation == "tanh" else (lambda x: x)
        )
        self._residual = in_dim == out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.dropout(self.ln(self.linear(x))))
        if self._residual:
            h = x + h
        return h


def _mlp_blocks_spec(config: ModelConfig) -> list[tuple[int, int, str, float]]:
    """(in_dim, out_dim, activation, dropout) per block. Last block: no dropout on logits."""
    flat = config.flatten_size
    hidden = config.n_hidden
    vocab = config.vocab_size
    n = config.num_mlp_layers
    drop = config.dropout
    spec: list[tuple[int, int, str, float]] = []
    in_d = flat
    for _ in range(n):
        spec.append((in_d, hidden, "tanh", drop))
        in_d = hidden
    spec.append((in_d, vocab, "identity", 0.0))
    return spec


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class CategoryConditionedNameModel(nn.Module):
    """
    Trunk: embed -> project to n_hidden -> for block in blocks: x = x + block(x, context).
    Head: flatten -> MLP blocks (residual only when dims match).
    Shortcuts are identity only; no learned shortcut projections.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__()
        self.config = config or ModelConfig(**kwargs)
        c = self.config

        self.token_embedding = nn.Embedding(c.vocab_size, c.n_embd)
        self.cat_embedding = nn.Embedding(c.cat_vocab_size, c.cat_emb_dim)

        # Single projection (no residual: dimension change n_embd -> n_hidden)
        self.project = nn.Linear(c.n_embd, c.n_hidden, bias=False)

        # Trunk: all blocks (B, T, n_hidden) -> (B, T, n_hidden); applied as x = x + block(x)
        cat_cross = CategoryCrossAttention(
            c.cat_emb_dim, c.n_hidden, c.head_size, c.dropout
        )
        self.blocks = nn.ModuleList([
            TrunkBlockWithContext(cat_cross),
            *[
                MultiHeadSelfAttention(c.n_hidden, c.num_heads, c.block_size, c.dropout)
                for _ in range(c.num_attention_blocks)
            ],
        ])

        self.mlp_blocks = nn.ModuleList([
            MLPBlock(in_d, out_d, drop, act)
            for in_d, out_d, act, drop in _mlp_blocks_spec(c)
        ])

        self._init_weights(c.last_layer_scale)

    def _init_weights(self, last_layer_scale: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        last_block = self.mlp_blocks[-1]
        last_block.ln.weight.data.mul_(last_layer_scale)
        last_block.ln.bias.data.mul_(last_layer_scale)

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

    @torch.no_grad()
    def generate(
        self,
        cat_idx: int | torch.Tensor,
        itos: dict[int, str],
        *,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> str:
        """
        Sample a single name for the given category.

        Args:
            cat_idx: scalar or (1,) category index
            itos: index-to-string mapping for decoding
            max_new_tokens: max characters (including '.' terminator)
            temperature: sampling temperature
            top_k: if set, only sample from top-k logits
            generator: optional RNG for reproducibility

        Returns:
            Decoded name string (without trailing '.')
        """
        self.eval()
        device = next(self.parameters()).device
        if isinstance(cat_idx, int):
            cat_idx = torch.tensor([cat_idx], device=device, dtype=torch.long)
        elif cat_idx.dim() == 0:
            cat_idx = cat_idx.unsqueeze(0).to(device)
        else:
            cat_idx = cat_idx.to(device)

        context = [0] * self.block_size
        has_space = False
        out: list[int] = []

        for _ in range(max_new_tokens):
            x = torch.tensor([context], device=device, dtype=torch.long)
            logits, _ = self.forward(x, cat_idx, targets=None)
            logits = logits[0] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1, generator=generator).item()

            if itos.get(ix, "") == " ":
                has_space = True
            if ix == 0:
                if not has_space:
                    ix = next(i for i, s in itos.items() if s == " ")
                    has_space = True
                else:
                    break
            context = context[1:] + [ix]
            out.append(ix)

        return "".join(itos.get(i, "") for i in out)
