from core.code_block_detector import detect_code_block, CodeBlockWatcher, CodeBlockDetection


class TestDetectCodeBlock:

    def test_normal_code_block(self):
        text = 'Some text\n```python\n# compute sum\nprint(2 + 2)\n```\nMore text'
        det = detect_code_block(text)
        assert det is not None
        assert det.executable == "print(2 + 2)"
        assert det.comments == ["# compute sum"]
        assert text[det.start:det.end] == '```python\n# compute sum\nprint(2 + 2)\n```'

    def test_no_comments(self):
        text = '```python\nprint("hello")\n```'
        det = detect_code_block(text)
        assert det is not None
        assert det.comments == []
        assert det.executable == 'print("hello")'

    def test_multiple_blocks_detects_first(self):
        text = '```python\nprint(1)\n```\ntext\n```python\nprint(2)\n```'
        det = detect_code_block(text)
        assert det is not None
        assert det.executable == "print(1)"

    def test_unclosed_block_returns_none(self):
        text = '```python\nprint(1)\n'
        det = detect_code_block(text)
        assert det is None

    def test_empty_code_block(self):
        text = '```python\n```'
        det = detect_code_block(text)
        assert det is not None
        assert det.executable == ""
        assert det.comments == []

    def test_code_block_with_only_comments(self):
        text = '```python\n# this is a comment\n# another comment\n```'
        det = detect_code_block(text)
        assert det is not None
        assert det.comments == ["# this is a comment", "# another comment"]
        assert det.executable == ""

    def test_multiple_comments_then_code(self):
        text = '```python\n# step 1\n# step 2\nx = 42\nprint(x)\n```'
        det = detect_code_block(text)
        assert det is not None
        assert det.comments == ["# step 1", "# step 2"]
        assert det.executable == "x = 42\nprint(x)"

    def test_inline_comment_not_treated_as_leading(self):
        text = '```python\nx = 1  # inline\nprint(x)\n```'
        det = detect_code_block(text)
        assert det is not None
        assert det.comments == []
        assert "x = 1  # inline" in det.executable

    def test_comment_after_code_not_leading(self):
        text = '```python\nprint(1)\n# this is after code\nprint(2)\n```'
        det = detect_code_block(text)
        assert det is not None
        assert det.comments == []
        assert "# this is after code" in det.executable

    def test_bare_triple_backtick_block(self):
        text = '```\nprint("hi")\n```'
        det = detect_code_block(text)
        assert det is not None
        assert det.executable == 'print("hi")'

    def test_py_language_tag(self):
        text = '```py\nprint(1)\n```'
        det = detect_code_block(text)
        assert det is not None

    def test_Python_capitalized_tag(self):
        text = '```Python\nprint(1)\n```'
        det = detect_code_block(text)
        assert det is not None

    def test_no_code_block_at_all(self):
        text = "Just plain text with no code blocks whatsoever."
        det = detect_code_block(text)
        assert det is None

    def test_backticks_in_prose_not_code_block(self):
        text = "Use `print(x)` to display the variable."
        det = detect_code_block(text)
        assert det is None

    def test_multiline_code(self):
        text = '```python\n# GDP calc\nimport math\nx = 3.05e12\nresult = x * 0.15\nprint(result)\n```'
        det = detect_code_block(text)
        assert det is not None
        assert det.comments == ["# GDP calc"]
        assert "import math" in det.executable
        assert "print(result)" in det.executable

    def test_start_end_positions(self):
        prefix = "I need to calculate something.\n"
        code = '```python\nprint(42)\n```'
        suffix = "\nThe answer is 42."
        text = prefix + code + suffix
        det = detect_code_block(text)
        assert det is not None
        assert det.start == len(prefix)
        assert det.end == len(prefix) + len(code)


class TestCodeBlockWatcher:

    def _feed_text(self, watcher, text):
        """Feed text token by token (character by character for testing)."""
        for ch in text:
            signal = watcher.feed_token(ch)
            if signal != "continue":
                return signal
        return "continue"

    def test_detects_complete_block(self):
        w = CodeBlockWatcher()
        text = 'Some text\n```python\nprint(1)\n```\nmore'
        signal = self._feed_text(w, text)
        assert signal == "code_block_complete"

    def test_no_block_stays_continue(self):
        w = CodeBlockWatcher()
        signal = self._feed_text(w, "Just plain text, no code.")
        assert signal == "continue"

    def test_eos_detected(self):
        w = CodeBlockWatcher()
        signal = self._feed_text(w, "Some text<|endoftext|>")
        assert signal == "eos"

    def test_unclosed_block_stays_continue(self):
        w = CodeBlockWatcher()
        signal = self._feed_text(w, "```python\nprint(1)\n")
        assert signal == "continue"
        assert w.state == "IN_CODE_BLOCK"

    def test_reset(self):
        w = CodeBlockWatcher()
        self._feed_text(w, "```python\nprint(1)\n")
        assert w.state == "IN_CODE_BLOCK"
        w.reset()
        assert w.state == "NORMAL"
        assert w.buffer == ""

    def test_get_detection_after_complete(self):
        w = CodeBlockWatcher()
        text = '```python\n# test\nprint(42)\n```'
        self._feed_text(w, text)
        det = w.get_detection()
        assert det is not None
        assert det.comments == ["# test"]
        assert det.executable == "print(42)"

    def test_custom_eos_token(self):
        w = CodeBlockWatcher(eos_token="<|im_end|>")
        signal = self._feed_text(w, "hello<|im_end|>")
        assert signal == "eos"

    def test_multi_token_feeding(self):
        """Feed realistic multi-character tokens."""
        w = CodeBlockWatcher()
        tokens = ["Some", " text", "\n```", "python", "\n", "print", "(", "1", ")", "\n", "```"]
        signals = []
        for tok in tokens:
            sig = w.feed_token(tok)
            signals.append(sig)
            if sig != "continue":
                break
        assert signals[-1] == "code_block_complete"
