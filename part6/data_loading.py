"""
Data loading and validation for fantasy name generation.

Handles Kaggle dataset download, CSV loading, string normalization,
character/category vocabularies, and dataset building with validation.
"""

from __future__ import annotations

import glob
import random
import re
from typing import NamedTuple

import pandas as pd
import torch


# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------


def normalize_string_df(df: pd.DataFrame, column: str | int = 0) -> pd.Series:
    """Normalize a string column: NFKD, ASCII, lowercase, strip punctuation/digits."""
    col = df[column] if isinstance(column, int) else df[column]
    return (
        col.astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\d+", "", regex=True)
        .str.replace("/", "")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace('"', "'")
        .str.replace("_", " ")
        .str.replace("`", "'")
        .str.replace(".", "")
        .str.strip()
    )


# ---------------------------------------------------------------------------
# Vocabulary types (for type hints and clarity)
# ---------------------------------------------------------------------------


class CharacterVocab(NamedTuple):
    stoi: dict[str, int]
    itos: dict[int, str]
    size: int


class CategoryVocab(NamedTuple):
    stoi: dict[str, int]
    itos: dict[int, str]
    size: int
    normalized_categories: list[str]


# ---------------------------------------------------------------------------
# Loading words and categories
# ---------------------------------------------------------------------------


def load_words_and_categories(
    kaggle_path: str,
    *,
    extra_csv_path: str | None = None,
    extra_name_column: str = "name",
    extra_category_column: str = "source_category",
) -> tuple[list[str], list[str]]:
    """
    Load words and categories from Kaggle CSV files and optional extra CSV.

    Returns:
        (words, categories) with aligned lengths.
    """
    names: list[pd.Series] = []
    categories: list[list[str]] = []

    for f in glob.glob(kaggle_path + "/*.csv"):
        df = pd.read_csv(f, header=None)
        series = normalize_string_df(df)
        names.append(series)
        match = re.search(r"[\w]*\.csv", f)
        if match:
            cat_name = match.group(0).replace(".csv", "")
            categories.append([cat_name] * len(series))
        else:
            categories.append([""] * len(series))

    words = pd.concat(names, ignore_index=True).tolist()
    categories_flat = [c for sub in categories for c in sub]

    if extra_csv_path:
        try:
            extra = pd.read_csv(extra_csv_path)
            series2 = normalize_string_df(extra, extra_name_column)
            cat2 = extra[extra_category_column].astype(str).str.replace("Category: ", "", regex=False)
            words.extend(series2.tolist())
            categories_flat.extend(cat2.tolist())
        except Exception as e:
            raise FileNotFoundError(f"Cannot load extra CSV {extra_csv_path}: {e}") from e

    return words, categories_flat


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------


def build_character_vocabulary(words: list[str]) -> CharacterVocab:
    """Build character vocab from words. Special token '.' has index 0."""
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return CharacterVocab(stoi=stoi, itos=itos, size=len(itos))


def build_category_vocabulary(categories: list[str]) -> CategoryVocab:
    """Normalize categories and build category vocab. Uses 'unknown' for empty/NaN."""
    normalized: list[str] = []
    for cat in categories:
        if pd.isna(cat) or str(cat).strip() == "":
            normalized.append("unknown")
        else:
            c = str(cat).lower().replace("category:", "").strip()
            normalized.append(c if c else "unknown")

    unique = sorted(set(normalized))
    stoi = {c: i for i, c in enumerate(unique)}
    itos = {i: c for c, i in stoi.items()}
    return CategoryVocab(stoi=stoi, itos=itos, size=len(stoi), normalized_categories=normalized)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------


def build_dataset(
    words: list[str],
    word_categories: list[str],
    char_vocab: CharacterVocab,
    cat_vocab: CategoryVocab,
    block_size: int,
    *,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (X, Y, C) tensors for next-character prediction with category per sample.

    X: (N, block_size) context indices
    Y: (N,) next character index
    C: (N,) category index for each sample
    """
    stoi = char_vocab.stoi
    cat_stoi = cat_vocab.stoi
    unknown_idx = cat_stoi.get("unknown", 0)

    X_list: list[list[int]] = []
    Y_list: list[int] = []
    C_list: list[int] = []

    missing_chars: set[str] = set()
    skipped_empty = 0
    skipped_bad_char = 0

    for idx, w in enumerate(words):
        if not w or not w.strip():
            skipped_empty += 1
            continue

        cat = word_categories[idx] if idx < len(word_categories) else "unknown"
        cat_idx = cat_stoi.get(cat, unknown_idx)

        context = [0] * block_size

        try:
            for ch in w + ".":
                if ch not in stoi:
                    missing_chars.add(ch)
                    skipped_bad_char += 1
                    raise KeyError(f"Character '{ch}' not in vocab")
                ix = stoi[ch]
                X_list.append(context.copy())
                Y_list.append(ix)
                C_list.append(cat_idx)
                context = context[1:] + [ix]
        except KeyError:
            continue

    if verbose:
        if missing_chars:
            print(f"  Warning: characters missing from vocab: {sorted(missing_chars)}")
        if skipped_empty:
            print(f"  Skipped {skipped_empty} empty words.")
        if not missing_chars and not skipped_bad_char:
            print("  ✓ All words processed successfully.")

    X = torch.tensor(X_list, dtype=torch.long)
    Y = torch.tensor(Y_list, dtype=torch.long)
    C = torch.tensor(C_list, dtype=torch.long)
    if verbose:
        print(f"  Shapes: X {X.shape}, Y {Y.shape}, C {C.shape}")
    return X, Y, C


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------


def get_train_val_test_splits(
    words: list[str],
    categories: list[str],  # must align with words; use cat_vocab.normalized_categories if you built cat_vocab from raw
    char_vocab: CharacterVocab,
    cat_vocab: CategoryVocab,
    block_size: int,
    *,
    train_frac: float = 0.95,
    val_frac: float = 0.03,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Shuffle words/categories together, split, and build X,Y,C for train/val/test.

    Returns:
        ( (Xtr, Ytr, Ctr), (Xdev, Ydev, Cdev), (Xte, Yte, Cte) )
    """
    pairs = list(zip(words, categories))
    random.seed(seed)
    random.shuffle(pairs)
    words_shuffled, cats_shuffled = zip(*pairs)
    words_shuffled = list(words_shuffled)
    cats_shuffled = list(cats_shuffled)

    n = len(words_shuffled)
    n1 = int(n * train_frac)
    n2 = int(n * (train_frac + val_frac))

    train_words, train_cats = words_shuffled[:n1], cats_shuffled[:n1]
    val_words, val_cats = words_shuffled[n1:n2], cats_shuffled[n1:n2]
    test_words, test_cats = words_shuffled[n2:], cats_shuffled[n2:]

    if verbose:
        print("Building train/val/test datasets...")
    Xtr, Ytr, Ctr = build_dataset(train_words, train_cats, char_vocab, cat_vocab, block_size, verbose=verbose)
    Xdev, Ydev, Cdev = build_dataset(val_words, val_cats, char_vocab, cat_vocab, block_size, verbose=verbose)
    Xte, Yte, Cte = build_dataset(test_words, test_cats, char_vocab, cat_vocab, block_size, verbose=verbose)

    if verbose:
        print(f"  Train: {len(Xtr)}, Val: {len(Xdev)}, Test: {len(Xte)}")
    return (Xtr, Ytr, Ctr), (Xdev, Ydev, Cdev), (Xte, Yte, Cte)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset(
    X: torch.Tensor,
    Y: torch.Tensor,
    C: torch.Tensor,
    char_vocab: CharacterVocab,
    cat_vocab: CategoryVocab,
    *,
    verbose: bool = True,
) -> bool:
    """
    Check for NaNs, index bounds, and alignment. Returns True if all checks pass.
    """
    vocab_size = char_vocab.size
    cat_size = cat_vocab.size
    ok = True

    if verbose:
        print("\n" + "=" * 60)
        print("DATASET VALIDATION")
        print("=" * 60)
        print(f"Vocab size: {vocab_size}")
        print(f"Shapes: X {X.shape}, Y {Y.shape}, C {C.shape}")

    if X.shape[0] != Y.shape[0] or Y.shape[0] != C.shape[0]:
        if verbose:
            print("  ✗ X, Y, C batch sizes do not match.")
        ok = False

    if torch.isnan(X).any() or torch.isnan(Y).any() or torch.isnan(C).any():
        if verbose:
            print("  ✗ NaNs found in X, Y, or C.")
        ok = False
    elif verbose:
        print("  No NaNs in X, Y, C.")

    x_min, x_max = X.min().item(), X.max().item()
    y_min, y_max = Y.min().item(), Y.max().item()
    c_min, c_max = C.min().item(), C.max().item()

    if x_min < 0 or x_max >= vocab_size or y_min < 0 or y_max >= vocab_size:
        if verbose:
            print(f"  ✗ X or Y indices out of range [0, {vocab_size - 1}].")
        ok = False
    elif verbose:
        print(f"  X indices in [0, {vocab_size - 1}], Y in [0, {vocab_size - 1}].")

    if c_min < 0 or c_max >= cat_size:
        if verbose:
            print(f"  ✗ C indices out of range [0, {cat_size - 1}].")
        ok = False
    elif verbose:
        print(f"  C indices in [0, {cat_size - 1}].")

    if verbose:
        print("=" * 60)
        print("✓ All validation checks passed." if ok else "✗ Some checks failed.")
        print("=" * 60 + "\n")
    return ok
