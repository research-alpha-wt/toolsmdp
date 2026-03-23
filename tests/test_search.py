"""Tests for search query extraction and pre-resolved search in executor."""

import pytest
from sandbox.executor import extract_search_query_strings, execute_code


class TestExtractSearchQueries:
    def test_double_quotes(self):
        assert extract_search_query_strings('search("France GDP")') == ["France GDP"]

    def test_single_quotes(self):
        assert extract_search_query_strings("search('France GDP')") == ["France GDP"]

    def test_multiple(self):
        code = 'search("q1")\nsearch("q2")'
        assert extract_search_query_strings(code) == ["q1", "q2"]

    def test_none(self):
        assert extract_search_query_strings("print(2+2)") == []

    def test_dedup(self):
        assert extract_search_query_strings('search("x")\nsearch("x")') == ["x"]


class TestPreResolvedSearch:
    def test_with_results(self):
        code = 'result = search("France GDP")\nprint(result)'
        out = execute_code(code, search_results={"France GDP": "3.05 trillion USD"})
        assert "3.05 trillion" in out

    def test_missing_query(self):
        code = 'result = search("unknown")\nprint(result)'
        out = execute_code(code, search_results={"other": "x"})
        assert "No results found" in out

    def test_placeholder_fallback(self):
        code = 'result = search("any")\nprint(result)'
        out = execute_code(code, search_enabled=True)
        assert "No search backend configured" in out

    def test_disabled(self):
        code = 'result = search("any")\nprint(result)'
        out = execute_code(code, search_enabled=False)
        assert "ERROR" in out

    def test_multiple(self):
        code = 'print(search("q1"))\nprint(search("q2"))'
        out = execute_code(code, search_results={"q1": "ans1", "q2": "ans2"})
        assert "ans1" in out and "ans2" in out


# Pyserini tests — skipped if not installed or JAVA_HOME not set
import os

HAS_PYSERINI = False
try:
    if os.environ.get("JAVA_HOME"):
        import pyserini
        HAS_PYSERINI = True
except ImportError:
    pass


@pytest.mark.skipif(not HAS_PYSERINI, reason="pyserini not installed")
class TestPyseriniSearch:
    def test_basic_query(self):
        from retrieval.search import get_search
        search = get_search()
        result = search("Freddie Mercury")
        assert "Mercury" in result

    def test_returns_ranked(self):
        from retrieval.search import get_search
        search = get_search()
        result = search("Albert Einstein", top_k=2)
        assert "[1]" in result
        assert "[2]" in result
