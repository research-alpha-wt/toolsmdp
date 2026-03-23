from core.code_block_detector import detect_code_block
from core.replacement import replace_code_block, replace_tool_output_with_context


class TestReplaceCodeBlock:

    def test_basic_replacement(self):
        text = 'I need the GDP.\n```python\n# GDP lookup\nresult = search("France GDP")\nprint(result)\n```\nNow I know.'
        det = detect_code_block(text)
        result = replace_code_block(text, det, "France GDP is $3.05 trillion.")
        assert "# GDP lookup" in result
        assert "France GDP is $3.05 trillion." in result
        assert "```python" not in result
        assert "search(" not in result
        assert "I need the GDP." in result
        assert "Now I know." in result

    def test_no_comments(self):
        text = '```python\nprint(2+2)\n```'
        det = detect_code_block(text)
        result = replace_code_block(text, det, "4")
        assert result == "4"
        assert "```" not in result

    def test_multiple_comments(self):
        text = '```python\n# step 1: find GDP\n# step 2: compute percentage\nx = 3.05e12 * 0.15\nprint(x)\n```'
        det = detect_code_block(text)
        result = replace_code_block(text, det, "457500000000.0")
        assert "# step 1: find GDP" in result
        assert "# step 2: compute percentage" in result
        assert "457500000000.0" in result
        assert "x = 3.05e12" not in result

    def test_error_replacement(self):
        text = '```python\n# GDP lookup\npritn("hello")\n```'
        det = detect_code_block(text)
        result = replace_code_block(text, det, "ERROR: NameError: name 'pritn' is not defined")
        assert "# GDP lookup" in result
        assert "ERROR:" in result
        assert "```" not in result

    def test_empty_stdout(self):
        text = '```python\n# assign variable\nx = 42\n```'
        det = detect_code_block(text)
        result = replace_code_block(text, det, "")
        assert "# assign variable" in result
        assert "```" not in result

    def test_context_reads_naturally(self):
        text = (
            "What is 15% of France's GDP?\n\n"
            "I need to find France's GDP first.\n"
            '```python\n# France GDP lookup\nresult = search("France GDP 2024")\nprint(result)\n```\n'
            "Now I can calculate the answer."
        )
        det = detect_code_block(text)
        result = replace_code_block(
            text, det, "France's GDP in 2024 was approximately $3.05 trillion."
        )
        # Should read like a natural flow
        assert "I need to find France's GDP first." in result
        assert "# France GDP lookup" in result
        assert "France's GDP in 2024 was approximately $3.05 trillion." in result
        assert "Now I can calculate the answer." in result
        assert "```" not in result

    def test_preserves_text_before_and_after(self):
        before = "Before the code block. "
        after = " After the code block."
        text = before + '```python\nprint(1)\n```' + after
        det = detect_code_block(text)
        result = replace_code_block(text, det, "1")
        assert result.startswith("Before the code block. ")
        assert result.endswith(" After the code block.")

    def test_multiline_stdout(self):
        text = '```python\nfor i in range(3):\n    print(i)\n```'
        det = detect_code_block(text)
        result = replace_code_block(text, det, "0\n1\n2")
        assert "0\n1\n2" in result
        assert "```" not in result

    def test_comments_and_no_stdout(self):
        text = '```python\n# this does nothing visible\nx = 1\n```'
        det = detect_code_block(text)
        result = replace_code_block(text, det, "")
        assert result == "# this does nothing visible"

    def test_only_stdout_no_comments(self):
        text = '```python\nprint("result")\n```'
        det = detect_code_block(text)
        result = replace_code_block(text, det, "result")
        assert result == "result"


class TestReplaceToolOutputWithContext:

    def test_basic_replacement(self):
        text = "Before.\nSearch returned: France GDP is $3.05 trillion.\nAfter."
        result = replace_tool_output_with_context(
            text, "Search returned: France GDP is $3.05 trillion.", "France GDP is $3T."
        )
        assert "France GDP is $3T." in result
        assert "Search returned:" not in result
        assert "Before." in result
        assert "After." in result

    def test_multiline_tool_output(self):
        tool_output = "Result 1: Paris\nResult 2: Lyon\nResult 3: Marseille"
        text = f"I searched.\n{tool_output}\nNow I know."
        result = replace_tool_output_with_context(text, tool_output, "Capital is Paris.")
        assert "Capital is Paris." in result
        assert "Result 1:" not in result

    def test_tool_output_not_found(self):
        text = "Some text without the tool output."
        result = replace_tool_output_with_context(text, "nonexistent output", "context")
        assert result == text

    def test_empty_context_content(self):
        text = "Before.\nTool output here.\nAfter."
        result = replace_tool_output_with_context(text, "Tool output here.", "")
        assert "Tool output here." not in result
        assert "Before.\n" in result
        assert "\nAfter." in result

    def test_only_first_occurrence_replaced(self):
        text = "AAA BBB AAA"
        result = replace_tool_output_with_context(text, "AAA", "CCC")
        assert result == "CCC BBB AAA"
