"""
Data loading and validation for fantasy name generation.

Handles Kaggle dataset download, CSV loading, string normalization,
character/category vocabularies, and dataset building with validation.
"""

from __future__ import annotations

import glob
import random
import re
from typing import Any, NamedTuple

import pandas as pd
import torch

from part7.category_groups import CATEGORY_GROUP_MAP


# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------


def normalize_string_df(df: pd.DataFrame, column: str | int = 0) -> pd.Series:
    """Normalize a string column: NFKD, ASCII, lowercase, strip punctuation/digits."""
    col = df[column] if isinstance(column, int) else df[column]
    return (
        col.astype(str)
        # .str.normalize("NFKD")
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


def load_dir(dir_path: str) -> tuple[list[str], list[str]]:
    """
    Load all CSV files in a directory. No header; category = filename (without .csv).

    Returns:
        (words, categories) with aligned lengths.
    """
    names: list[pd.Series] = []
    categories: list[list[str]] = []

    for f in glob.glob(dir_path.rstrip("/\\") + "/*.csv"):
        df = pd.read_csv(f, header=None)
        series = normalize_string_df(df)
        names.append(series)
        match = re.search(r"[\w]*\.csv", f)
        if match:
            cat_name = match.group(0).replace(".csv", "")
            categories.append([cat_name] * len(series))
        else:
            categories.append([""] * len(series))

    words = pd.concat(names, ignore_index=True).tolist() if names else []
    categories_flat = [c for sub in categories for c in sub]
    return words, categories_flat


def load_file(
    file_path: str,
    name_column: str,
    category_column: str,
    *,
    category_prefix: str = "Category: ",
) -> tuple[list[str], list[str]]:
    """
    Load a single CSV with header. Uses name_column and category_column.

    Returns:
        (words, categories) with aligned lengths.
    """
    df = pd.read_csv(file_path)
    series = normalize_string_df(df, name_column)
    cat = (
        df[category_column]
        .astype(str)
        .str.replace(category_prefix, "", regex=False)
        .str.strip()
    )
    words = series.tolist()
    categories_flat = cat.tolist()
    return words, categories_flat


def load_words_and_categories(
    *,
    dirs_configs: list[dict] | None = None,
    files_configs: list[dict] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Load and merge words/categories from multiple dirs and/or files by config.

    Each config dict is unpacked into load_dir or load_file. Example::

        dirs_configs = [{"dir_path": "/path/to/kaggle"}]
        files_configs = [
            {"file_path": "data/extra.csv", "name_column": "name", "category_column": "source_category"},
        ]
        words, categories = load_words_and_categories(dirs_configs=dirs_configs, files_configs=files_configs)

    Args:
        dirs_configs: List of kwargs for load_dir (e.g. {"dir_path": "..."}).
        files_configs: List of kwargs for load_file (e.g. {"file_path": "...", "name_column": "...", "category_column": "..."}).

    Returns:
        (words, categories) with aligned lengths.
    """
    all_words: list[str] = []
    all_categories: list[str] = []

    for config in (dirs_configs or []):
        w, c = load_dir(**config)
        all_words.extend(w)
        all_categories.extend(c)

    for config in (files_configs or []):
        w, c = load_file(**config)
        all_words.extend(w)
        all_categories.extend(c)

    return all_words, all_categories


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


def build_category_vocabulary(
    categories: list[str],
    *,
    use_category_groups: bool = True,
) -> CategoryVocab:
    """
    Normalize categories and build category vocab. Uses 'unknown' for empty/NaN.
    If use_category_groups is True, maps known categories to grouped labels via CATEGORY_GROUP_MAP.
    """
    normalized: list[str] = []
    for cat in categories:
        if pd.isna(cat) or str(cat).strip() == "":
            normalized.append("unknown")
        else:
            c = str(cat).lower().replace("category:", "").strip()
            c = c if c else "unknown"
            if use_category_groups:
                c = CATEGORY_GROUP_MAP.get(c, c)
            normalized.append(c)

    unique = sorted(set(normalized))
    stoi = {c: i for i, c in enumerate(unique)}
    itos = {i: c for c, i in stoi.items()}
    return CategoryVocab(stoi=stoi, itos=itos, size=len(stoi), normalized_categories=normalized)


# ---------------------------------------------------------------------------
# Dataset building (character-level and token-level)
# ---------------------------------------------------------------------------


def build_dataset_tokens(
    words: list[str],
    word_categories: list[str],
    tokenizer: Any,  # BPETokenizer: .encode(str), .pad_token_id, .end_token_id, .stoi, .itos, .size
    cat_vocab: CategoryVocab,
    block_size: int,
    *,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (X, Y, C) tensors for next-token prediction with category per sample.

    tokenizer must have: encode(text) -> list[int], pad_token_id, end_token_id, and .stoi/.itos/.size.
    Context is padded with pad_token_id. Each word is encoded and appended with end_token_id.
    Produces one (context, next_token, category) per position in each word's token sequence.

    X: (N, block_size) context token indices (padded with <PAD>)
    Y: (N,) next token index
    C: (N,) category index for each sample
    """
    cat_stoi = cat_vocab.stoi
    unknown_idx = cat_stoi.get("unknown", 0)
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.end_token_id

    X_list: list[list[int]] = []
    Y_list: list[int] = []
    C_list: list[int] = []
    skipped_empty = 0
    skipped_no_tokens = 0

    for idx, w in enumerate(words):
        if not w or not w.strip():
            skipped_empty += 1
            continue

        cat = word_categories[idx] if idx < len(word_categories) else "unknown"
        cat_idx = cat_stoi.get(cat, unknown_idx)

        token_ids = tokenizer.encode(w) + [end_id]
        if not token_ids:
            skipped_no_tokens += 1
            continue

        context = [pad_id] * block_size
        for tid in token_ids:
            X_list.append(context.copy())
            Y_list.append(tid)
            C_list.append(cat_idx)
            context = context[1:] + [tid]

    if verbose:
        if skipped_empty:
            print(f"  Skipped {skipped_empty} empty words.")
        if skipped_no_tokens:
            print(f"  Skipped {skipped_no_tokens} words with no tokens.")
        print(f"  Shapes: X {len(X_list)} x {block_size}, Y {len(Y_list)}, C {len(C_list)}")

    X = torch.tensor(X_list, dtype=torch.long)
    Y = torch.tensor(Y_list, dtype=torch.long)
    C = torch.tensor(C_list, dtype=torch.long)
    if verbose and X.numel():
        print(f"  Tensors: X {X.shape}, Y {Y.shape}, C {C.shape}")
    return X, Y, C


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


def get_train_val_test_splits_tokens(
    words: list[str],
    categories: list[str],
    tokenizer: Any,
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
    Shuffle words/categories, split, and build token-level X,Y,C for train/val/test.
    Uses build_dataset_tokens. Returns same structure as get_train_val_test_splits.
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
        print("Building train/val/test datasets (token-level)...")
    Xtr, Ytr, Ctr = build_dataset_tokens(
        train_words, train_cats, tokenizer, cat_vocab, block_size, verbose=verbose
    )
    Xdev, Ydev, Cdev = build_dataset_tokens(
        val_words, val_cats, tokenizer, cat_vocab, block_size, verbose=verbose
    )
    Xte, Yte, Cte = build_dataset_tokens(
        test_words, test_cats, tokenizer, cat_vocab, block_size, verbose=verbose
    )

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
