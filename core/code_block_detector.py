import re
from dataclasses import dataclass


@dataclass
class CodeBlockDetection:
    code: str
    comments: list[str]
    executable: str
    start: int
    end: int


# Matches ```python ... ``` blocks (with optional language tag variations)
_CODE_BLOCK_PATTERN = re.compile(
    r"(```(?:python|py|Python)?\s*\n)(.*?)(```)",
    re.DOTALL,
)


def detect_code_block(text: str) -> CodeBlockDetection | None:
    """Find the first complete ```python code block in text.

    Returns None if no complete code block is found.
    """
    match = _CODE_BLOCK_PATTERN.search(text)
    if match is None:
        return None

    opening_fence = match.group(1)
    code_body = match.group(2)
    start = match.start()
    end = match.end()

    comments = []
    executable_lines = []
    for line in code_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and not executable_lines:
            comments.append(stripped)
        else:
            executable_lines.append(line)

    executable = "\n".join(executable_lines).strip()

    return CodeBlockDetection(
        code=code_body.strip(),
        comments=comments,
        executable=executable,
        start=start,
        end=end,
    )


class CodeBlockWatcher:
    """Real-time state machine that watches tokens during generation.

    Feed tokens one at a time. Returns a signal:
      - "continue": keep generating
      - "code_block_complete": closing ``` detected, stop and execute
      - "eos": end-of-sequence token detected
    """

    def __init__(self, eos_token: str = "<|endoftext|>"):
        self.state = "NORMAL"
        self.buffer = ""
        self.eos_token = eos_token
        self._fence_buffer = ""

    def reset(self):
        self.state = "NORMAL"
        self.buffer = ""
        self._fence_buffer = ""

    def feed_token(self, token_text: str) -> str:
        self.buffer += token_text

        if self.eos_token in self.buffer:
            return "eos"

        if self.state == "NORMAL":
            if self._check_opening_fence():
                self.state = "IN_CODE_BLOCK"
            return "continue"

        elif self.state == "IN_CODE_BLOCK":
            if self._check_closing_fence():
                self.state = "NORMAL"
                return "code_block_complete"
            return "continue"

        return "continue"

    def _check_opening_fence(self) -> bool:
        return bool(re.search(r"```(?:python|py|Python)?\s*\n", self.buffer))

    def _check_closing_fence(self) -> bool:
        # After the opening fence, look for a closing ``` on its own line
        # Split buffer at the opening fence to only search the code body
        parts = re.split(r"```(?:python|py|Python)?\s*\n", self.buffer, maxsplit=1)
        if len(parts) < 2:
            return False
        code_body = parts[1]
        # Closing fence: ``` at start of a line (possibly after whitespace)
        return bool(re.search(r"\n\s*```", code_body))

    def get_detection(self) -> CodeBlockDetection | None:
        return detect_code_block(self.buffer)
