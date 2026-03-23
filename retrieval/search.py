"""Wikipedia search via Pyserini BM25.

Uses the pre-built wikipedia-dpr-100w Lucene index (21M passages).
First run downloads ~9.2 GB. Cached at ~/.cache/pyserini/ after that.

Requires: pip install pyserini faiss-cpu
Requires: Java 11+ (set JAVA_HOME)

Usage:
    from retrieval.search import get_search
    search = get_search()
    print(search("France military spending"))
"""

import json

_searcher = None


def get_search():
    """Load the Pyserini searcher (singleton). Returns a callable: query(str, top_k) -> str."""
    global _searcher
    if _searcher is not None:
        return _searcher

    from pyserini.search.lucene import LuceneSearcher

    print("Loading Wikipedia search index (first run downloads ~9.2 GB)...")
    searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")
    print(f"Loaded. {searcher.num_docs} passages.")

    def query(q: str, top_k: int = 3) -> str:
        hits = searcher.search(q, k=top_k)
        if not hits:
            return "No relevant results found."
        parts = []
        for rank, hit in enumerate(hits, 1):
            doc = json.loads(searcher.doc(hit.docid).raw())
            title = doc.get("title", "")
            text = doc.get("contents", "")
            parts.append(f"[{rank}] {title}\n{text}")
        return "\n\n".join(parts)

    _searcher = query
    return _searcher
