import re
import csv
from urllib.request import urlopen
from html import unescape

URL = "https://archive.org/stream/story_games_name_project/the_story_games_names_project_djvu.txt"

# --- 1) Download ---
html = urlopen(URL).read().decode("utf-8", errors="ignore")
html = unescape(html)

print(html[:100])

# The IA "stream" page is HTML; we want the text content.
# This crude stripper is usually enough for this page; BS4 is nicer, but optional.
text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
text = re.sub(r"(?is)<.*?>", "\n", text)  # tags -> newlines
lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
lines = [ln for ln in lines if ln]  # drop empty

# --- 2) Extract "name-like" entries from numbered lists ---
# Patterns we see in this document:
#  - "1)" on its own line, then name on next non-empty line
#  - "1) Name" on the same line
#  - "1 . Name" / "1. Name" (some sections OCR this way)
#  - OCR confusions: "I)" (1), "T)" (7) etc. We'll accept single letters too.
marker_only = re.compile(r"^(?:\d{1,2}|[IT])\)\s*$")          # e.g. "9)" or "I)"
marker_inline = re.compile(r"^(?:\d{1,2}|[IT])\)\s*(.+)$")   # e.g. "2) Hasad"
dot_inline    = re.compile(r"^\d{1,2}\s*[\.\)]\s*(.+)$")     # e.g. "1 . Prudence" / "2. Rachel ..."

raw = []

i = 0
while i < len(lines):
    ln = lines[i]

    m = marker_inline.match(ln)
    if m:
        raw.append(m.group(1).strip())
        i += 1
        continue

    m = dot_inline.match(ln)
    if m:
        raw.append(m.group(1).strip())
        i += 1
        continue

    if marker_only.match(ln):
        # take next non-empty line as the entry
        if i + 1 < len(lines):
            raw.append(lines[i + 1].strip())
            i += 2
            continue

    i += 1

# --- 3) Clean ---
def clean_entry(s: str) -> str:
    s = s.strip()

    # If the entry includes an explanation "Name - blah" or "Name / blah", keep only the name part
    s = re.split(r"\s+[-/]\s+", s, maxsplit=1)[0].strip()

    # Remove ALL non-word, non-space characters (except apostrophes) from anywhere in the string
    # This removes parentheses, backslashes, forward slashes, hyphens, etc.
    s = re.sub(r"[^\w\s'']", "", s).strip()

    # Remove leading stray punctuation (in case anything remains)
    s = re.sub(r"^[^\w]+", "", s).strip()

    # Drop anything with digits (usually not a name)
    if re.search(r"\d", s):
        return ""

    # Drop very long "entries" that are clearly not names (tune if you want)
    if len(s) > 60:
        return ""

    # Drop entries that are obviously list sentences (common in OCR)
    if "," in s or ";" in s:
        return ""

    return s

cleaned = []
seen = set()
for s in raw:
    c = clean_entry(s)
    if not c:
        continue

    key = c.casefold()
    if key not in seen:
        seen.add(key)
        cleaned.append(c)

# --- 4) Write outputs ---
with open("names_raw.txt", "w", encoding="utf-8") as f:
    for s in raw:
        f.write(s + "\n")

with open("names_clean.txt", "w", encoding="utf-8") as f:
    for s in cleaned:
        f.write(s + "\n")

with open("names_clean.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["name"])
    for s in cleaned:
        w.writerow([s])

print(f"Raw extracted: {len(raw)}")
print(f"Clean deduped: {len(cleaned)}")
print("Wrote: names_raw.txt, names_clean.txt, names_clean.csv")