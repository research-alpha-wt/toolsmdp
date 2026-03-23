from core.context_block_detector import (
    detect_context_block,
    ContextBlockWatcher,
    ContextBlockDetection,
)


class TestDetectContextBlock:

    def test_basic_context_block(self):
        text = "Some text\n<context>The answer is 42.</context>\nMore text"
        det = detect_context_block(text)
        assert det is not None
        assert det.content == "The answer is 42."
        assert text[det.start:det.end] == "<context>The answer is 42.</context>"

    def test_multiline_content(self):
        text = "<context>Line 1\nLine 2\nLine 3</context>"
        det = detect_context_block(text)
        assert det is not None
        assert det.content == "Line 1\nLine 2\nLine 3"

    def test_no_closing_tag_returns_none(self):
        text = "<context>This block never closes"
        det = detect_context_block(text)
        assert det is None

    def test_no_context_block_at_all(self):
        text = "Just plain text with no context blocks."
        det = detect_context_block(text)
        assert det is None

    def test_empty_content(self):
        text = "<context></context>"
        det = detect_context_block(text)
        assert det is not None
        assert det.content == ""

    def test_whitespace_content_stripped(self):
        text = "<context>  spaced content  </context>"
        det = detect_context_block(text)
        assert det is not None
        assert det.content == "spaced content"

    def test_multiple_blocks_detects_first(self):
        text = "<context>first</context> then <context>second</context>"
        det = detect_context_block(text)
        assert det is not None
        assert det.content == "first"

    def test_start_end_positions(self):
        prefix = "I looked it up.\n"
        block = "<context>France GDP is $3T.</context>"
        suffix = "\nSo the answer is..."
        text = prefix + block + suffix
        det = detect_context_block(text)
        assert det is not None
        assert det.start == len(prefix)
        assert det.end == len(prefix) + len(block)

    def test_math_context_block(self):
        text = "<context>347 * 28 = 9716</context>"
        det = detect_context_block(text)
        assert det is not None
        assert det.content == "347 * 28 = 9716"

    def test_search_context_block(self):
        text = (
            "The search returned many results.\n"
            "<context>The capital of France is Paris, with a population of 2.1 million.</context>\n"
            "Now I can answer."
        )
        det = detect_context_block(text)
        assert det is not None
        assert "capital of France is Paris" in det.content

    def test_angle_brackets_in_prose_not_context(self):
        text = "Use <b>bold</b> for emphasis."
        det = detect_context_block(text)
        assert det is None


class TestContextBlockWatcher:

    def _feed_text(self, watcher, text):
        """Feed text character by character."""
        for ch in text:
            signal = watcher.feed_token(ch)
            if signal != "continue":
                return signal
        return "continue"

    def test_detects_complete_block(self):
        w = ContextBlockWatcher()
        text = "Some text\n<context>Result is 42.</context>\nmore"
        signal = self._feed_text(w, text)
        assert signal == "context_block_complete"

    def test_no_block_stays_continue(self):
        w = ContextBlockWatcher()
        signal = self._feed_text(w, "Just plain text, no context.")
        assert signal == "continue"

    def test_eos_detected(self):
        w = ContextBlockWatcher()
        signal = self._feed_text(w, "Some text<|endoftext|>")
        assert signal == "eos"

    def test_eos_inside_block_detected(self):
        w = ContextBlockWatcher()
        signal = self._feed_text(w, "<context>partial<|endoftext|>")
        assert signal == "eos"

    def test_unclosed_block_stays_continue(self):
        w = ContextBlockWatcher()
        signal = self._feed_text(w, "<context>This never closes")
        assert signal == "continue"
        assert w.state == "IN_CONTEXT_BLOCK"

    def test_reset(self):
        w = ContextBlockWatcher()
        self._feed_text(w, "<context>partial content")
        assert w.state == "IN_CONTEXT_BLOCK"
        w.reset()
        assert w.state == "NORMAL"
        assert w.buffer == ""

    def test_get_detection_after_complete(self):
        w = ContextBlockWatcher()
        self._feed_text(w, "<context>The answer is 42.</context>")
        det = w.get_detection()
        assert det is not None
        assert det.content == "The answer is 42."

    def test_custom_eos_token(self):
        w = ContextBlockWatcher(eos_token="<|im_end|>")
        signal = self._feed_text(w, "hello<|im_end|>")
        assert signal == "eos"

    def test_budget_exceeded(self):
        w = ContextBlockWatcher(max_tokens=5)
        # Feed opening tag first
        for ch in "<context>":
            w.feed_token(ch)
        # Now feed tokens inside the block — each char is a "token" here
        signals = []
        for ch in "abcdefghij":
            sig = w.feed_token(ch)
            signals.append(sig)
            if sig != "continue":
                break
        assert signals[-1] == "budget_exceeded"

    def test_closing_before_budget(self):
        w = ContextBlockWatcher(max_tokens=100)
        text = "<context>short</context>"
        signal = self._feed_text(w, text)
        assert signal == "context_block_complete"

    def test_multi_token_feeding(self):
        """Feed realistic multi-character tokens."""
        w = ContextBlockWatcher()
        tokens = ["Some", " text", "\n<", "context", ">", "The result", " is", " 42", ".", "</", "context", ">"]
        signals = []
        for tok in tokens:
            sig = w.feed_token(tok)
            signals.append(sig)
            if sig != "continue":
                break
        assert signals[-1] == "context_block_complete"

    def test_state_transitions(self):
        w = ContextBlockWatcher()
        assert w.state == "NORMAL"
        for ch in "<context>":
            w.feed_token(ch)
        assert w.state == "IN_CONTEXT_BLOCK"
        for ch in "content</context>":
            sig = w.feed_token(ch)
            if sig != "continue":
                break
        assert w.state == "NORMAL"
