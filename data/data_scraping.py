import csv
import time
import requests
from urllib.parse import urljoin

WIKI_BASE = "https://forgottenrealms.fandom.com/"
API = urljoin(WIKI_BASE, "api.php")  # Fandom MediaWiki Action API endpoint
ROOT_CATEGORY = "Category:Inhabitants_by_race"

OUT_CSV = "forgotten_realms_inhabitants_by_race.csv"
OUT_TXT = "forgotten_realms_inhabitants_by_race.txt"

SLEEP_S = 0.5          # be nice
MAX_RETRIES = 5
TIMEOUT = 30


def api_get(params: dict) -> dict:
    """Polite GET with retries for transient failures."""
    params = dict(params)
    params.setdefault("format", "json")
    # Helpful for some MediaWiki installs; harmless if ignored:
    params.setdefault("formatversion", "2")

    # Print what we're requesting
    action = params.get("action", "unknown")
    list_type = params.get("list", "unknown")
    cmtitle = params.get("cmtitle", "unknown")
    cmtype = params.get("cmtype", "unknown")
    print(f"  → API Request: action={action}, list={list_type}, category={cmtitle}, type={cmtype}")

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(API, params=params, timeout=TIMEOUT, headers={
                "User-Agent": "name-scraper/1.0 (personal use; polite; contact: none)"
            })
            r.raise_for_status()
            result = r.json()
            # Print success
            if attempt > 1:
                print(f"    ✓ Success on attempt {attempt}")
            return result
        except Exception as e:
            last_err = e
            print(f"    ✗ Attempt {attempt} failed: {e}")
            # exponential backoff
            time.sleep(SLEEP_S * attempt * 2)

    raise RuntimeError(f"API request failed after {MAX_RETRIES} retries: {last_err}")


def list_category_members(cat_title: str, cmtype: str):
    """
    Yield category members for a category.
    cmtype: "page" or "subcat"
    """
    cont = None
    page_num = 0
    total_items = 0
    
    while True:
        page_num += 1
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat_title,
            "cmtype": cmtype,      # "page" or "subcat"
            "cmlimit": "500",
        }
        if cmtype == "page":
            params["cmnamespace"] = "0"  # main/article namespace only

        if cont:
            params["cmcontinue"] = cont
            print(f"    (page {page_num}, continuing from {cont[:20]}...)")

        data = api_get(params)
        items = data.get("query", {}).get("categorymembers", [])
        batch_count = len(items)
        total_items += batch_count
        
        if page_num == 1:
            print(f"    Found {batch_count} items in first batch", end="")
        else:
            print(f"    Found {batch_count} more items (total so far: {total_items})", end="")
        
        for item in items:
            yield item

        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            print(f"    ✓ Complete: {total_items} total {cmtype}s")
            break

        time.sleep(SLEEP_S)


def scrape_inhabitants(root_category: str):
    """
    BFS through subcategories starting at root_category.
    Collect all page titles (names) in each category.
    """
    visited_cats = set()
    queue = [root_category]

    rows = []  # dicts: {"name": ..., "source_category": ...}
    
    print(f"\nStarting BFS scrape from root category: {root_category}")
    print("=" * 70)

    while queue:
        cat = queue.pop(0)
        if cat in visited_cats:
            print(f"\n⚠ Skipping already visited category: {cat}")
            continue
        
        visited_cats.add(cat)
        print(f"\n[{len(visited_cats)}] Processing category: {cat}")
        print(f"    Queue size: {len(queue)} categories remaining")

        # 1) collect pages (names) directly in this category
        print(f"    Fetching PAGES from {cat}...")
        page_count = 0
        for item in list_category_members(cat, cmtype="page"):
            title = item.get("title", "").strip()
            if not title:
                continue
            rows.append({"name": title, "source_category": cat})
            page_count += 1
        print(f"    ✓ Collected {page_count} pages from {cat}")
        print(f"    Total names collected so far: {len(rows)}")

        # 2) enqueue subcategories
        print(f"    Fetching SUBCATEGORIES from {cat}...")
        subcat_count = 0
        for item in list_category_members(cat, cmtype="subcat"):
            subcat = item.get("title", "").strip()
            if subcat and subcat.startswith("Category:") and subcat not in visited_cats:
                queue.append(subcat)
                subcat_count += 1
        print(f"    ✓ Found {subcat_count} new subcategories to process")

        time.sleep(SLEEP_S)
    
    print("\n" + "=" * 70)
    print(f"BFS complete! Processed {len(visited_cats)} categories, collected {len(rows)} names")
    return rows


def dedupe_rows(rows):
    """Dedupe by name (case-insensitive), keeping first occurrence."""
    seen = set()
    out = []
    for r in rows:
        key = r["name"].casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


if __name__ == "__main__":
    print(f"Configuration:")
    print(f"  API endpoint: {API}")
    print(f"  Root category: {ROOT_CATEGORY}")
    print(f"  Sleep between requests: {SLEEP_S}s")
    print(f"  Max retries: {MAX_RETRIES}")
    print(f"  Timeout: {TIMEOUT}s")
    
    rows = scrape_inhabitants(ROOT_CATEGORY)
    
    print(f"\nBefore deduplication: {len(rows)} names")
    rows = dedupe_rows(rows)
    print(f"After deduplication: {len(rows)} unique names")

    # Write CSV (with provenance)
    print(f"\nWriting CSV to {OUT_CSV}...")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "source_category"])
        w.writeheader()
        w.writerows(rows)
    print(f"✓ Wrote: {OUT_CSV}")

    # Write TXT (name-only)
    print(f"Writing TXT to {OUT_TXT}...")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["name"] + "\n")
    print(f"✓ Wrote: {OUT_TXT}")

    print(f"\n{'='*70}")
    print(f"Done! Final count: {len(rows)} unique names")
    print(f"Files written: {OUT_CSV}, {OUT_TXT}")