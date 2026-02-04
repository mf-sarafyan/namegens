"""
Generation logic: mixin class for models that support token/name generation.

- GenerationMixin: provides generate_token and generate_name; inherit to add generation to a model.
- Standalone generate_token / generate_name: thin wrappers for non-inheritance use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GenerationMixin:
    """
    Mixin that adds generate_token and generate_name to a model.

    The model must provide: block_size (int), forward(idx, cat_idx, targets=None),
    parameters(), and eval(). Typically used with nn.Module: class MyModel(GenerationMixin, nn.Module).
    """

    def _cat_idx_tensor(self, cat_idx: int | torch.Tensor) -> torch.Tensor:
        """Normalize cat_idx to a (1,) tensor on the model's device."""
        device = next(self.parameters()).device
        if isinstance(cat_idx, int):
            return torch.tensor([cat_idx], device=device, dtype=torch.long)
        if isinstance(cat_idx, torch.Tensor):
            t = cat_idx.unsqueeze(0).to(device) if cat_idx.dim() == 0 else cat_idx.to(device)
            return t
        return torch.tensor([int(cat_idx)], device=device, dtype=torch.long)

    @torch.no_grad()
    def generate_token(
        self,
        context: list[int],
        cat_idx: int | torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> int:
        """
        Sample the next token index given a context of length block_size.

        Args:
            context: list of token indices, length must equal self.block_size.
            cat_idx: category index (scalar or 1-D tensor).
            temperature: sampling temperature.
            top_k: if set, sample only from top-k logits.
            generator: optional RNG for reproducibility.

        Returns:
            Next token index (int).
        """
        self.eval()
        device = next(self.parameters()).device
        block_size = self.block_size
        if len(context) != block_size:
            raise ValueError(f"context length must be {block_size}, got {len(context)}")

        cat_idx_t = self._cat_idx_tensor(cat_idx)
        x = torch.tensor([context], device=device, dtype=torch.long)
        logits, _ = self(x, cat_idx_t, targets=None)
        logits = logits[0] / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).item()

    @torch.no_grad()
    def generate_name(
        self,
        cat_idx: int | torch.Tensor,
        itos: dict[int, str],
        stoi: dict[str, int],
        *,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        replace_end_with: str | None = " ",
        end_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> str:
        """
        Generate a full name (one or more words) for the given category.

        When the model would output the end token (<EOS>), replace_end_with controls
        what happens:
        - None: stop immediately (single word, no space).
        - " ": output a space and continue (multiple words).
        - " the ": output " the " and continue (e.g. "Gimli the Bold").

        Args:
            cat_idx: category index.
            itos: index-to-string for decoding.
            stoi: string-to-index for encoding replace_end_with.
            max_new_tokens: max characters to generate.
            temperature: sampling temperature.
            top_k: if set, sample only from top-k logits.
            generator: optional RNG.
            replace_end_with: string to output instead of ending on first end token;
                encoded with stoi and appended so generation continues. None = stop on first end.
            end_token_id: token id that ends the sequence (e.g. <EOS>). Default 0 for backward compat.
            pad_token_id: token id used to fill context (e.g. <PAD>). Default 0 for backward compat.

        Returns:
            Decoded name string (no trailing end token).
        """
        self.eval()
        device = next(self.parameters()).device
        block_size = self.block_size
        cat_idx_t = self._cat_idx_tensor(cat_idx)
        eos_id = end_token_id if end_token_id is not None else 0
        pad_id = pad_token_id if pad_token_id is not None else 0

        replacement_tokens: list[int] = []
        if replace_end_with is not None and replace_end_with:
            replacement_tokens = [stoi[c] for c in replace_end_with if c in stoi]

        context = [pad_id] * block_size
        out: list[int] = []
        replaced_end = False

        for _ in range(max_new_tokens):
            ix = self.generate_token(
                context, cat_idx_t, temperature=temperature, top_k=top_k, generator=generator
            )

            if ix == eos_id:
                if replace_end_with is not None and not replaced_end and replacement_tokens:
                    out.extend(replacement_tokens)
                    context = context[len(replacement_tokens) :] + replacement_tokens
                    replaced_end = True
                else:
                    break
            elif ix != pad_id:
                out.append(ix)
                context = context[1:] + [ix]
            else:
                context = context[1:] + [ix]

        return "".join(itos.get(i, "") for i in out)


# ---------------------------------------------------------------------------
# Standalone wrappers (for use without inheritance: generate_token(model, ...))
# ---------------------------------------------------------------------------


def generate_token(
    model: GenerationMixin,
    context: list[int],
    cat_idx: int | torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Thin wrapper: model.generate_token(context, cat_idx, ...)."""
    return model.generate_token(
        context, cat_idx, temperature=temperature, top_k=top_k, generator=generator
    )


def generate_name(
    model: GenerationMixin,
    cat_idx: int | torch.Tensor,
    itos: dict[int, str],
    stoi: dict[str, int],
    *,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
    replace_end_with: str | None = " ",
    end_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> str:
    """Thin wrapper: model.generate_name(cat_idx, itos, stoi, ...)."""
    return model.generate_name(
        cat_idx,
        itos,
        stoi,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
        replace_end_with=replace_end_with,
        end_token_id=end_token_id,
        pad_token_id=pad_token_id,
    )
