from dataclasses import dataclass, field


@dataclass
class Segment:
    """One generation segment in a trajectory.

    The model generates from start_context until it emits a code fence (tool_call)
    or EOS. This dataclass stores everything the training loop needs for that segment.
    The model never sees this object — it only sees start_context as a plain string.
    """
    start_context: str
    generated_text: str
    generated_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    termination: str = "eos"  # "tool_call" | "eos" | "truncated"
    tool_code: str | None = None
    tool_comments: list[str] | None = None
    tool_output: str | None = None

    # Filled during advantage computation
    advantage: float | None = None
    value_estimate: float | None = None  # V(s_k) at this segment's start
    value_target: float | None = None    # Bootstrap target for critic


@dataclass
class Trajectory:
    """Complete rollout for one question.

    Contains all segments and metadata for training and logging.
    The model sees full_context (clean text). Everything else is bookkeeping.
    """
    segments: list[Segment] = field(default_factory=list)
    full_context: str = ""
    reward: float | None = None

    @property
    def total_tool_calls(self) -> int:
        return sum(1 for s in self.segments if s.termination == "tool_call")

    @property
    def hit_segment_limit(self) -> bool:
        return (
            len(self.segments) > 0
            and self.segments[-1].termination == "truncated"
        )

    @property
    def num_segments(self) -> int:
        return len(self.segments)
