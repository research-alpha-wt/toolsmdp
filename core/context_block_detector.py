import re
from dataclasses import dataclass


@dataclass
class ContextBlockDetection:
    content: str
    start: int
    end: int


_CONTEXT_BLOCK_PATTERN = re.compile(
    r"<context>(.*?)</context>",
    re.DOTALL,
)


def detect_context_block(text: str) -> ContextBlockDetection | None:
    """Find the first complete <context>...</context> block in text."""
    match = _CONTEXT_BLOCK_PATTERN.search(text)
    if match is None:
        return None

    return ContextBlockDetection(
        content=match.group(1).strip(),
        start=match.start(),
        end=match.end(),
    )


class ContextBlockWatcher:
    """Real-time state machine that watches tokens for <context>...</context>.

    Feed tokens one at a time. Returns a signal:
      - "continue": keep generating
      - "context_block_complete": closing </context> detected
      - "eos": end-of-sequence token detected
      - "budget_exceeded": assimilation token budget hit while inside block
    """

    def __init__(self, eos_token: str = "<|endoftext|>", max_tokens: int = 256):
        self.state = "NORMAL"
        self.buffer = ""
        self.eos_token = eos_token
        self.max_tokens = max_tokens
        self._tokens_in_block = 0

    def reset(self):
        self.state = "NORMAL"
        self.buffer = ""
        self._tokens_in_block = 0

    def feed_token(self, token_text: str) -> str:
        self.buffer += token_text

        if self.eos_token in self.buffer:
            return "eos"

        if self.state == "NORMAL":
            if "<context>" in self.buffer:
                self.state = "IN_CONTEXT_BLOCK"
                self._tokens_in_block = 0
            return "continue"

        elif self.state == "IN_CONTEXT_BLOCK":
            self._tokens_in_block += 1

            if "</context>" in self.buffer:
                self.state = "NORMAL"
                return "context_block_complete"

            if self._tokens_in_block >= self.max_tokens:
                self.state = "NORMAL"
                return "budget_exceeded"

            return "continue"

        return "continue"

    def get_detection(self) -> ContextBlockDetection | None:
        return detect_context_block(self.buffer)
