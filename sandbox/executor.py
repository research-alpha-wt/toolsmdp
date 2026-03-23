import json
import re
import subprocess
import sys
import textwrap
import tempfile
import os

BLOCKED_PATTERNS = frozenset({
    "import os", "import sys", "import subprocess", "import shutil",
    "import socket", "import requests", "import urllib", "import importlib",
    "import pathlib", "import io", "from os", "from sys",
    "from subprocess", "from shutil", "from socket", "from requests",
    "from urllib", "from importlib", "from pathlib", "from io",
    "__import__", "exec(", "eval(", "compile(", "open(",
    "breakpoint(", "globals(", "locals(",
})

TIMEOUT_SECONDS = 5

_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "socket",
    "requests", "urllib", "importlib", "pathlib", "io",
    "signal", "ctypes", "multiprocessing", "threading",
})

_IMPORT_GUARD = textwrap.dedent("""\
    import builtins
    _orig_import = builtins.__import__
    _BLOCKED = {blocked}
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split('.')[0]
        if level == 0 and top in _BLOCKED:
            raise ImportError(f"Import '{{name}}' is not allowed")
        return _orig_import(name, globals, locals, fromlist, level)
    builtins.__import__ = _safe_import
""")

_SEARCH_PLACEHOLDER = textwrap.dedent("""\
    def search(query):
        return f"[Search results for: {query}] No search backend configured."
""")


def _build_search_func(search_results: dict[str, str]) -> str:
    """Build a search() function that returns pre-resolved results."""
    results_json = json.dumps(search_results)
    return textwrap.dedent(f"""\
        import json as _json
        _SEARCH_RESULTS = _json.loads('''{results_json}''')
        def search(query):
            return _SEARCH_RESULTS.get(query, "No results found for: " + query)
    """)


def extract_search_queries(code: str) -> list[str]:
    """Extract search() query strings from code. Handles search("...") and search('...')."""
    return re.findall(r'''search\(\s*(?:"([^"]+)"|'([^']+)')\s*\)''', code)


def extract_search_query_strings(code: str) -> list[str]:
    """Return deduplicated list of search query strings found in code."""
    queries = []
    for match in extract_search_queries(code):
        q = match[0] or match[1]  # one of the two groups matched
        if q and q not in queries:
            queries.append(q)
    return queries


def execute_code(
    code: str,
    search_enabled: bool = False,
    search_results: dict[str, str] | None = None,
    timeout: int = TIMEOUT_SECONDS,
) -> str:
    """Execute Python code in a sandboxed subprocess. Returns stdout or 'ERROR: ...'.

    If search_results is provided, search() calls use pre-resolved results.
    If search_enabled=True but no search_results, a placeholder is used.
    """
    if not code.strip():
        return ""

    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return f"ERROR: Blocked operation: {pattern}"

    script = _IMPORT_GUARD.format(blocked=repr(_BLOCKED_MODULES))
    if search_results:
        script += _build_search_func(search_results)
    elif search_enabled:
        script += _SEARCH_PLACEHOLDER
    script += "\n" + code

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
        )

        if result.returncode != 0:
            for line in reversed(result.stderr.strip().splitlines()):
                if line and not line.startswith(" ") and not line.startswith("Traceback"):
                    return f"ERROR: {line}"
            return "ERROR: Unknown error"

        return result.stdout.rstrip("\n")

    except subprocess.TimeoutExpired:
        return "ERROR: Execution timed out"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass
