"""
Byte-Pair Encoding (BPE) tokenizer for name generation.

Learns a subword vocabulary from a corpus; target vocab size (e.g. 128).
Reserves index 0 for <PAD> and index 1 for <EOS> (end-of-sequence).
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"


class BPETokenizer:
    """
    BPE tokenizer with a fixed target vocabulary size.

    - Index 0 is reserved for <PAD> (padding).
    - Index 1 is reserved for <EOS> (end-of-sequence).
    - Base vocab = [<PAD>, <EOS>] + corpus characters, then merge pairs until vocab_size is reached.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        pad_token: str = PAD_TOKEN,
        eos_token: str = EOS_TOKEN,
    ):
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.eos_token = eos_token
        self._vocab: dict[str, int] = {}
        self._merges: list[tuple[str, str]] = []  # merge order for encoding

    def fit(self, corpus: list[str]) -> BPETokenizer:
        """
        Learn BPE merges from the corpus. Call this before encode/decode.

        Args:
            corpus: list of strings (e.g. words).

        Returns:
            self for chaining.
        """
        # Base vocab: <PAD> (0), <EOS> (1), then all chars that appear (no special tokens in corpus)
        all_chars = set()
        for s in corpus:
            if s:
                all_chars.update(s)
        for special in (self.pad_token, self.eos_token, "."):
            all_chars.discard(special)
        base_tokens = [self.pad_token, self.eos_token] + sorted(all_chars)
        self._vocab = {t: i for i, t in enumerate(base_tokens)}
        self._merges = []

        # Represent corpus as list of token lists (each token is a single char string)
        word_tokens = [[c for c in w] for w in corpus if w]

        # Merge until we reach vocab_size
        while len(self._vocab) < self.vocab_size:
            # Count adjacent pairs
            pair_counts: Counter[tuple[str, str]] = Counter()
            for tokens in word_tokens:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += 1
            if not pair_counts:
                break
            best_pair = pair_counts.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self._vocab)
            self._vocab[new_token] = new_id
            self._merges.append(best_pair)

            # Replace all occurrences of best_pair with new_token
            for tokens in word_tokens:
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i + 1]) == best_pair:
                        tokens[i : i + 2] = [new_token]
                    else:
                        i += 1

        return self

    def encode(self, text: str) -> list[int]:
        """
        Encode a string to token ids.

        Does not append the EOS token; caller typically does encode(s) + [self.end_token_id].
        """
        if not self._vocab:
            raise RuntimeError("Tokenizer not fitted. Call fit(corpus) first.")
        tokens = list(text)
        for left, right in self._merges:
            new_token = left + right
            if new_token not in self._vocab:
                continue
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == left and tokens[i + 1] == right:
                    tokens[i : i + 2] = [new_token]
                else:
                    i += 1
        return [self._vocab[t] for t in tokens if t in self._vocab]

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        """
        Decode a list of token ids to a string.

        If skip_special is True (default), <PAD> and <EOS> are omitted from the output.
        """
        if not self._vocab:
            raise RuntimeError("Tokenizer not fitted. Call fit(corpus) first.")
        inv = {i: t for t, i in self._vocab.items()}
        if skip_special:
            pad_id = self.pad_token_id
            eos_id = self.end_token_id
            return "".join(inv.get(i, "") for i in ids if i != pad_id and i != eos_id)
        return "".join(inv.get(i, "") for i in ids)

    @property
    def pad_token_id(self) -> int:
        """Token id for padding (<PAD>)."""
        return self._vocab.get(self.pad_token, 0)

    @property
    def end_token_id(self) -> int:
        """Token id for end-of-sequence (<EOS>)."""
        return self._vocab.get(self.eos_token, 1)

    @property
    def stoi(self) -> dict[str, int]:
        """String to index mapping."""
        return dict(self._vocab)

    @property
    def itos(self) -> dict[int, str]:
        """Index to string mapping."""
        return {i: t for t, i in self._vocab.items()}

    @property
    def size(self) -> int:
        """Vocabulary size (for compatibility with CharacterVocab)."""
        return len(self._vocab)


def tokenizer_vocab(tokenizer: BPETokenizer) -> TokenizerVocab:
    """
    Return a CharacterVocab-like NamedTuple for use with dataset builders
    that expect .stoi, .itos, .size.
    """
    return TokenizerVocab(stoi=tokenizer.stoi, itos=tokenizer.itos, size=tokenizer.size)


class TokenizerVocab(NamedTuple):
    """Compatibility view: stoi, itos, size (like CharacterVocab)."""
    stoi: dict[str, int]
    itos: dict[int, str]
    size: int
