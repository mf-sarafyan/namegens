"""
Expand names from names_llm.csv into full names by category.

- If a category has both first-name subcategories (e.g. MALE, FEMALE) and a
  subcategory that represents surnames (subcategory containing "surname"), we
  form the cartesian product: each first name × each surname → "First Surname".
- Other categories can get specific logic later.

Usage:
  python data/multiply_names.py [--input data/names_llm.csv] [--output data/names_multiplied.csv]
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------
# Data loading and grouping
# -----------------------------------------------------------------------------

def load_names_csv(path: str | Path) -> list[dict]:
    """Load name,category,subcategory CSV; return list of dicts with keys name, category, subcategory.
    Skips empty rows and duplicate header rows.
    """
    path = Path(path)
    rows: list[dict] = []
    seen_header = False
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, fieldnames=("name", "category", "subcategory")):
            name = (row.get("name") or "").strip()
            category = (row.get("category") or "").strip()
            subcategory = (row.get("subcategory") or "").strip()
            if not name and not category and not subcategory:
                continue
            if name.lower() == "name" and category.lower() == "category" and subcategory.lower() == "subcategory":
                continue
            rows.append({"name": name, "category": category, "subcategory": subcategory})
    return rows


def group_by_category(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by category. Returns dict: category -> list of row dicts (name, category, subcategory)."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["category"]:
            by_cat[r["category"]].append(r)
    return dict(by_cat)


def group_by_subcategory(rows: list[dict]) -> dict[str, list[str]]:
    """Group names by subcategory within a category. Returns dict: subcategory -> list of names."""
    by_sub: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        sub = r.get("subcategory", "").strip() or "(blank)"
        name = (r.get("name") or "").strip()
        if name:
            by_sub[sub].append(name)
    return dict(by_sub)


# -----------------------------------------------------------------------------
# Surname detection and first-name vs surname split
# ------------------------------------------------------------------------------

def subcategory_contains_surnames(subcategory: str) -> bool:
    """True if this subcategory represents surnames (e.g. SURNAME, SURNAMES, FEMALE SURNAMES)."""
    if not subcategory:
        return False
    return "surname" in subcategory.lower()


def get_surname_subcategory_keys(names_by_sub: dict[str, list[str]]) -> list[str]:
    """Return subcategory keys that are considered surname lists."""
    return [k for k in names_by_sub if subcategory_contains_surnames(k)]


def get_first_name_subcategory_keys(names_by_sub: dict[str, list[str]]) -> list[str]:
    """Return subcategory keys that are first names (not surnames). Used for cartesian product."""
    return [k for k in names_by_sub if not subcategory_contains_surnames(k)]


# -----------------------------------------------------------------------------
# Composing full names
# ------------------------------------------------------------------------------

def cartesian_first_surnames(
    first_names: list[str],
    surnames: list[str],
    *,
    separator: str = " ",
) -> list[str]:
    """Cartesian product: each first name + separator + each surname. Deduplicated and sorted."""
    seen: set[str] = set()
    for fn in first_names:
        fn = fn.strip()
        if not fn:
            continue
        for sn in surnames:
            sn = sn.strip()
            if not sn:
                continue
            full = f"{fn}{separator}{sn}"
            seen.add(full)
    return sorted(seen)


def collect_all_first_names(names_by_sub: dict[str, list[str]], first_name_sub_keys: list[str]) -> list[str]:
    """Collect all names from the given first-name subcategories, deduplicated."""
    seen: set[str] = set()
    for k in first_name_sub_keys:
        for name in names_by_sub.get(k, []):
            n = name.strip()
            if n:
                seen.add(n)
    return sorted(seen)


def collect_all_surnames(names_by_sub: dict[str, list[str]], surname_sub_keys: list[str]) -> list[str]:
    """Collect all surnames from the given surname subcategories, deduplicated."""
    seen: set[str] = set()
    for k in surname_sub_keys:
        for name in names_by_sub.get(k, []):
            n = name.strip()
            if n:
                seen.add(n)
    return sorted(seen)


def build_full_names_for_category(
    category_name: str,
    names_by_sub: dict[str, list[str]],
    *,
    separator: str = " ",
) -> list[tuple[str, str]]:
    """
    For one category, build (full_name, category) pairs.

    - If the category has both first-name subcategories and surname subcategories,
      returns cartesian product of (first names × surnames) with category.
    - Otherwise returns (name, category) for each name, unchanged.

    Returns: list of (full_name, category).
    """
    surname_keys = get_surname_subcategory_keys(names_by_sub)
    first_keys = get_first_name_subcategory_keys(names_by_sub)

    if surname_keys and first_keys:
        first_names = collect_all_first_names(names_by_sub, first_keys)
        surnames = collect_all_surnames(names_by_sub, surname_keys)
        if not first_names or not surnames:
            # fallback: no product, just emit singles
            all_names = collect_all_first_names(names_by_sub, first_keys) + collect_all_surnames(names_by_sub, surname_keys)
            return [(n, category_name) for n in all_names]
        full_names = cartesian_first_surnames(first_names, surnames, separator=separator)
        return [(fn, category_name) for fn in full_names]

    # No surname subcategory: emit all names as-is with category
    result: list[tuple[str, str]] = []
    for names in names_by_sub.values():
        for n in names:
            n = n.strip()
            if n:
                result.append((n, category_name))
    return result


# -----------------------------------------------------------------------------
# Main: iterate categories and write output
# ------------------------------------------------------------------------------

def run(
    input_path: str | Path = "data/names_llm.csv",
    output_path: str | Path = "data/names_multiplied.csv",
    separator: str = " ",
) -> None:
    """Load CSV, multiply names by category, write full_name,category CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = load_names_csv(input_path)
    by_category = group_by_category(rows)

    output_rows: list[dict] = []
    for cat, cat_rows in sorted(by_category.items()):
        names_by_sub = group_by_subcategory(cat_rows)
        pairs = build_full_names_for_category(cat, names_by_sub, separator=separator)
        for full_name, category in pairs:
            output_rows.append({"full_name": full_name, "category": category})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["full_name", "category"])
        w.writeheader()
        w.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Expand names into full names by category (first × surname where applicable).")
    p.add_argument("--input", default="data/names_llm.csv", help="Input CSV (name, category, subcategory)")
    p.add_argument("--output", default="data/names_multiplied.csv", help="Output CSV (full_name, category)")
    p.add_argument("--separator", default=" ", help="Separator between first name and surname")
    args = p.parse_args()
    run(input_path=args.input, output_path=args.output, separator=args.separator)


if __name__ == "__main__":
    main()
