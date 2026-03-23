from core.code_block_detector import CodeBlockDetection


def replace_code_block(text: str, detection: CodeBlockDetection, stdout: str) -> str:
    """Replace a code block with its comments and execution output.

    The code block (including fences) is removed. In its place:
      1. Any leading comment lines from the code are preserved.
      2. The stdout from execution is appended.

    This produces natural-looking text where the model's intent annotation
    (comments) and the tool's result (stdout) remain, but the executable
    code vanishes.
    """
    parts = []

    if detection.comments:
        parts.append("\n".join(detection.comments))

    if stdout:
        parts.append(stdout)

    replacement = "\n".join(parts)

    return text[:detection.start] + replacement + text[detection.end:]


def replace_tool_output_with_context(text: str, tool_output: str, context_content: str) -> str:
    """Phase-2 replacement: swap raw tool output for distilled <context> content.

    After the model writes <context>key fact</context>, the raw stdout from
    the tool call is replaced with just the distilled content. Both the code
    AND the raw output vanish — only the model's summary remains.
    """
    idx = text.find(tool_output)
    if idx == -1:
        return text
    return text[:idx] + context_content + text[idx + len(tool_output):]
