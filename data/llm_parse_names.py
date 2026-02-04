"""
LLM-based extraction of names with categories and sub-categories from the Story Games
Names Project text. Iterates over chunks of the HTML-derived text and sends each to
an LLM. Supports:

  - Local Llama (Ollama): free, runs on your machine. Recommended for large runs.
  - OpenRouter: cloud, uses src/secrets.py OPENROUTER_API_KEY.

Usage:
  pip install openai
  python data/llm_parse_names.py

Local Llama (Ollama):
  1. Install Ollama: https://ollama.com
  2. Pull a model:   ollama pull llama3.2
  3. List models:   ollama list
  4. Run script:     python data/llm_parse_names.py --local
  Optionally:        python data/llm_parse_names.py --local --model llama3.1

Output: names_llm.csv with columns name, category, subcategory
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

from urllib.request import urlopen
from html import unescape

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Install openai: pip install openai")

# Resolve project root and load secrets (OPENROUTER_API_KEY)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.secrets import OPENROUTER_API_KEY
except ImportError:
    OPENROUTER_API_KEY = ""

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OLLAMA_BASE = "http://localhost:11434/v1"  # Ollama OpenAI-compatible API

# Same source as get_names.py
URL = "https://archive.org/stream/story_games_name_project/the_story_games_names_project_djvu.txt"

# Start: detect by content so we skip TOC/foreword and start at first name-list section
START_MARKER = "Because you, Shakespeare, and I know, there's nothing"  # start of "1001 Nights" (first section); if OCR differs, try "1001 Nights" or "These names are taken"
START_CHAR_FALLBACK = 80_000  # used only if START_MARKER not found in text
# End: character limit (~where the book ends)
END_CHAR = 550_000

# Chunk size (chars) and overlap: only this many chars are carried into the next prompt (minimal overlap)
CHUNK_CHARS = 5000
CHUNK_OVERLAP = 100

# Rate limit: seconds between API calls (avoid throttling)
API_DELAY_SEC = 0.5

# Models and output file
DEFAULT_MODEL = "openai/gpt-4o-mini"   # OpenRouter
DEFAULT_MODEL_LOCAL = "llama3.2"       # Ollama (use "ollama list" to see installed models)
OUTPUT_CSV = "names_llm.csv"


def fetch_and_strip_html() -> list[str]:
    """Download URL and strip to non-empty text lines (same logic as get_names.py)."""
    raw = urlopen(URL).read().decode("utf-8", errors="ignore")
    raw = unescape(raw)
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<.*?>", "\n", text)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def find_content_start(full_text: str, marker: str, fallback: int) -> int:
    """Find character position where real content starts (after TOC/formatting). Uses marker string, else fallback."""
    pos = full_text.find(marker)
    if pos != -1:
        return pos
    return min(fallback, len(full_text))


def make_chunks(lines: list[str], start_char: int = 0, end_char: int | None = None) -> list[str]:
    """Split lines into overlapping text chunks. start_char/end_char slice the full text before chunking."""
    if start_char > 0 or end_char is not None:
        full = "\n".join(lines)
        if start_char >= len(full):
            return []
        full = full[start_char:end_char] if end_char is not None else full[start_char:]
        if not full:
            return []
        lines = full.split("\n")
    chunks: list[str] = []
    buffer: list[str] = []
    length = 0
    for ln in lines:
        line_len = len(ln) + 1
        if length + line_len > CHUNK_CHARS and buffer:
            chunk_text = "\n".join(buffer)
            chunks.append(chunk_text)
            # Minimal overlap: keep only the last CHUNK_OVERLAP chars for the next prompt
            while buffer and len("\n".join(buffer)) > CHUNK_OVERLAP:
                buffer.pop(0)
            length = sum(len(x) + 1 for x in buffer)
        buffer.append(ln)
        length += line_len
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


SYSTEM_PROMPT = """You are extracting personal names (and only names suitable for characters) from a scanned book of name lists. The text has OCR errors and inconsistent formatting.

For each chunk of text:
- Identify the current SECTION or LIST name (e.g. "1001 Nights", "Amazons", "Angels and Demons"). That is the category.
- Identify the SUBCATEGORY when present (e.g. "MALE", "FEMALE", "MALE NAMES", "FEMALE NAMES", "SURNAMES", "PLACES", "GODS"). If there is no clear subcategory, use empty string or the same as category.
- Extract entries that can be used as PERSON NAMES. Be sure to include and identify first and last names and split by gender (first names, surnames, full names). Job titles and places can be used as last names as well - be sure to keep them in, identified as last names. Do NOT include: name origins and descriptions,food items, clothing, forensic techniques, etc. When in doubt, include it and we can filter later.
- Output strictly as CSV with header: name,category,subcategory
- One row per name. Use double quotes if a field contains a comma. No other commentary or markdown."""

USER_PROMPT_TEMPLATE = """Extract all person names with their category and subcategory from this text chunk. Output CSV only (name,category,subcategory).

Text chunk:
---
{chunk}
---"""


def parse_llm_csv_response(response: str) -> list[tuple[str, str, str]]:
    """Parse model output into (name, category, subcategory) rows. Tolerates markdown code blocks."""
    response = response.strip()
    # Remove markdown code block if present
    if "```" in response:
        start = response.find("```")
        end = response.find("```", start + 3)
        if end != -1:
            response = response[start + 3 : end].strip()
        else:
            response = response[start + 3 :].strip()
    if response.startswith("csv"):
        response = response[3:].strip()
    rows: list[tuple[str, str, str]] = []
    for line in response.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name,"):
            if line and "name" in line.lower() and "category" in line.lower():
                continue  # skip header
            if not line:
                continue
        # Simple CSV parse (no quoted commas in our data usually)
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 3:
            rows.append((parts[0], parts[1], parts[2]))
        elif len(parts) == 2:
            rows.append((parts[0], parts[1], ""))
        elif len(parts) == 1 and parts[0]:
            rows.append((parts[0], "", ""))
    return rows


def call_llm(client: OpenAI, model: str, chunk: str, *, debug: bool = False) -> list[tuple[str, str, str]]:
    """Send one chunk to the LLM and return parsed (name, category, subcategory) list."""
    user = USER_PROMPT_TEMPLATE.format(chunk=chunk)
    if debug:
        print("\n" + "=" * 60 + " INPUT (system prompt) " + "=" * 60)
        print(SYSTEM_PROMPT[:500] + "..." if len(SYSTEM_PROMPT) > 500 else SYSTEM_PROMPT)
        print("\n" + "=" * 60 + " INPUT (user prompt / chunk) " + "=" * 60)
        print(user)
        print("=" * 60 + "\n")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    text = resp.choices[0].message.content or ""
    if debug:
        print("=" * 60 + " OUTPUT (raw model response) " + "=" * 60)
        print(text)
        print("=" * 60 + "\n")
    return parse_llm_csv_response(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract names with categories from Story Games Names Project via LLM.")
    p.add_argument("--local", action="store_true", help="Use local Ollama instead of OpenRouter")
    p.add_argument("--model", type=str, default=None, help="Model name (default: gpt-4o-mini for OpenRouter, llama3.2 for Ollama)")
    p.add_argument("--debug", action="store_true", help="Print each request's input (prompts + chunk) and raw model output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.local:
        base_url = OLLAMA_BASE
        api_key = "ollama"  # Ollama doesn't validate; any value is fine
        model = args.model or DEFAULT_MODEL_LOCAL
        print(f"Using local Llama (Ollama): {model}")
        print("  Tip: run 'ollama list' to see installed models; 'ollama pull <name>' to add one.")
    else:
        if not OPENROUTER_API_KEY:
            raise SystemExit(
                "Set OPENROUTER_API_KEY in src/secrets.py, or use local: python data/llm_parse_names.py --local"
            )
        base_url = OPENROUTER_BASE
        api_key = OPENROUTER_API_KEY
        model = args.model or DEFAULT_MODEL
        print(f"Using OpenRouter: {model}")

    script_dir = _SCRIPT_DIR
    os.chdir(script_dir)

    print(f"Fetching and stripping HTML from {URL} ...")
    lines = fetch_and_strip_html()
    print(f"Got {len(lines)} non-empty lines.")

    full_text = "\n".join(lines)
    start_char = find_content_start(full_text, START_MARKER, START_CHAR_FALLBACK)
    print(f"Content start: char {start_char} (marker {START_MARKER!r} found)" if full_text.find(START_MARKER) != -1 else f"Content start: char {start_char} (fallback, marker not found)")

    chunks = make_chunks(lines, start_char=start_char, end_char=END_CHAR)
    print(f"Split into {len(chunks)} chunks (chars {start_char}–{END_CHAR}, max ~{CHUNK_CHARS} chars, overlap {CHUNK_OVERLAP}).")
    if args.debug:
        print("Debug mode: will print input/output for each chunk.\n")

    client = OpenAI(base_url=base_url, api_key=api_key)
    seen: set[tuple[str, str, str]] = set()
    out_path = script_dir / OUTPUT_CSV

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "category", "subcategory"])

        for i, chunk in enumerate(chunks):
            print(f"Chunk {i + 1}/{len(chunks)} ... ", end="", flush=True)
            try:
                rows = call_llm(client, model, chunk, debug=args.debug)
                added = 0
                for r in rows:
                    key = (r[0].strip(), r[1].strip(), r[2].strip())
                    if not key[0]:
                        continue
                    if key not in seen:
                        seen.add(key)
                        w.writerow(key)
                        added += 1
                print(f"got {len(rows)} rows, {added} new.")
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(API_DELAY_SEC)

    print(f"Done. Total {len(seen)} rows in {out_path}")


if __name__ == "__main__":
    main()
