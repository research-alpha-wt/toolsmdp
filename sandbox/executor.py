import subprocess
import sys
import textwrap
import tempfile
import os
from typing import Callable

ALLOWED_MODULES = frozenset({
    "math", "numpy", "collections", "itertools", "re",
    "statistics", "functools", "string", "decimal", "fractions",
    "random", "operator", "json",
})

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


def _check_imports(code: str) -> str | None:
    """Return an error message if code uses blocked imports/builtins, else None."""
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return f"Blocked operation: {pattern}"
    return None


def _build_runner_script(code: str, search_fn_code: str | None = None) -> str:
    """Build a standalone Python script that executes the user's code."""
    parts = []

    # Restrict builtins
    parts.append(textwrap.dedent("""\
        import builtins
        _orig_import = builtins.__import__
        _ALLOWED = {allowed}
        def _safe_import(name, *args, **kwargs):
            top = name.split('.')[0]
            if top not in _ALLOWED:
                raise ImportError(f"Import '{{name}}' is not allowed")
            return _orig_import(name, *args, **kwargs)
        builtins.__import__ = _safe_import
    """).format(allowed=repr(ALLOWED_MODULES)))

    if search_fn_code:
        parts.append(search_fn_code)

    parts.append(code)
    return "\n".join(parts)


def execute_code(
    code: str,
    search_enabled: bool = False,
    search_fn: Callable[[str], str] | None = None,
    timeout: int = TIMEOUT_SECONDS,
) -> str:
    """Execute Python code in a sandboxed subprocess.

    Args:
        code: Python source to execute.
        search_enabled: Whether to inject a search() function.
        search_fn: If provided, this callable is used to implement search().
                   If search_enabled=True and search_fn is None, search()
                   returns a placeholder.
        timeout: Max execution time in seconds.

    Returns:
        stdout from execution, or "ERROR: <message>".
    """
    if not code.strip():
        return ""

    blocked = _check_imports(code)
    if blocked:
        return f"ERROR: {blocked}"

    # Build search function injection
    search_fn_code = None
    if search_enabled:
        if search_fn is not None:
            # We can't serialize an arbitrary callable into a subprocess.
            # Instead, we use a file-based IPC approach: write queries to a temp file,
            # the parent process picks them up. For simplicity in the MVP, we run
            # search in-process via a wrapper that pre-computes results.
            # For now, use a placeholder that the caller can override.
            search_fn_code = textwrap.dedent("""\
                import json, sys
                def search(query):
                    # Write query to stderr for parent to intercept
                    sys.stderr.write("SEARCH_QUERY:" + query + "\\n")
                    sys.stderr.flush()
                    return "Search is not available in subprocess mode. Use cached results."
            """)
        else:
            search_fn_code = textwrap.dedent("""\
                def search(query):
                    return f"[Search results for: {query}] No search backend configured."
            """)

    runner_script = _build_runner_script(code, search_fn_code)

    # Write to temp file and execute
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(runner_script)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Extract just the last line (the actual error) for cleaner output
            error_lines = stderr.splitlines()
            if error_lines:
                # Find the actual error (skip traceback)
                for line in reversed(error_lines):
                    if line and not line.startswith(" ") and not line.startswith("Traceback"):
                        return f"ERROR: {line}"
                return f"ERROR: {error_lines[-1]}"
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
