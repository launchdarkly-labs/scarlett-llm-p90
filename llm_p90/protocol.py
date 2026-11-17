"""Pre-registered execution choices for the benchmark and experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentProtocol:
    runner_region: str
    concurrency: int
    streaming: bool | None
    timeout_seconds: int
    temperature: float | None
    max_tokens: int
    prompt_caching: bool
    warmup_rows: int
    retry_429: bool

    def __post_init__(self) -> None:
        if not self.runner_region.strip():
            raise ValueError("runner_region is required")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.streaming is None:
            raise ValueError("streaming must be explicitly true or false")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.warmup_rows < 0:
            raise ValueError("warmup_rows cannot be negative")


@dataclass
class TrialState:
    max_attempts: int = 2
    attempts_started: int = 0

    def begin_attempt(self) -> int:
        if self.attempts_started >= self.max_attempts:
            raise RuntimeError("attempt budget exhausted")
        self.attempts_started += 1
        return self.attempts_started
