"""Interactive search test script.

Try queries against the Wikipedia search index.
Edit the QUERIES list below to test whatever you want.

Usage:
    python scripts/try_search.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.search import get_search

# ──────────────────────────────────────────────
# EDIT THESE QUERIES — add/remove whatever you want
# ──────────────────────────────────────────────
QUERIES = [
    "Freddie Mercury death year",
    "who wrote Crazy Little Thing Called Love",
    "France military spending",
    "capital of Japan",
    "Albert Einstein Nobel Prize",
]

TOP_K = 3  # number of results per query


def main():
    search = get_search()

    for query in QUERIES:
        print(f"Q: {query}")
        print("-" * 60)
        print(search(query, top_k=TOP_K))
        print()
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
