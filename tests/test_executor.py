from sandbox.executor import execute_code


class TestExecuteCode:

    def test_simple_arithmetic(self):
        assert execute_code("print(2 + 2)") == "4"

    def test_string_operation(self):
        assert execute_code('print("hello"[::-1])') == "olleh"

    def test_multiline_output(self):
        code = "for i in range(3):\n    print(i)"
        assert execute_code(code) == "0\n1\n2"

    def test_name_error(self):
        result = execute_code("pritn(42)")
        assert result.startswith("ERROR:")
        assert "NameError" in result

    def test_syntax_error(self):
        result = execute_code("def f(:")
        assert result.startswith("ERROR:")

    def test_zero_division(self):
        result = execute_code("print(1/0)")
        assert result.startswith("ERROR:")
        assert "ZeroDivision" in result

    def test_timeout(self):
        result = execute_code("while True: pass", timeout=1)
        assert result.startswith("ERROR:")
        assert "timed out" in result.lower()

    def test_allowed_import_math(self):
        result = execute_code("import math\nprint(round(math.pi, 2))")
        assert result == "3.14"

    def test_allowed_import_collections(self):
        code = "from collections import Counter\nprint(Counter('aab'))"
        result = execute_code(code)
        assert "a" in result and "2" in result

    def test_allowed_import_json(self):
        code = 'import json\nprint(json.dumps({"a": 1}))'
        result = execute_code(code)
        assert '"a": 1' in result

    def test_blocked_import_os(self):
        result = execute_code("import os\nprint(os.getcwd())")
        assert result.startswith("ERROR:")

    def test_blocked_import_subprocess(self):
        result = execute_code("import subprocess")
        assert result.startswith("ERROR:")

    def test_blocked_open(self):
        result = execute_code("open('/etc/passwd')")
        assert result.startswith("ERROR:")

    def test_blocked_eval(self):
        result = execute_code("eval('1+1')")
        assert result.startswith("ERROR:")

    def test_empty_code(self):
        assert execute_code("") == ""
        assert execute_code("   ") == ""

    def test_no_output(self):
        result = execute_code("x = 42")
        assert result == ""

    def test_search_disabled_by_default(self):
        result = execute_code("search('test')")
        assert result.startswith("ERROR:")

    def test_search_enabled_placeholder(self):
        result = execute_code("print(search('France GDP'))", search_enabled=True)
        assert "France GDP" in result

    def test_large_output(self):
        code = "for i in range(1000):\n    print(i)"
        result = execute_code(code)
        assert "999" in result

    def test_float_precision(self):
        result = execute_code("print(0.15 * 3.05e12)")
        assert "457500000000" in result
