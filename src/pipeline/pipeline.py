from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol


class PipelineObserver(Protocol):
    def on_step_start(self, step_name: str, context: "PipelineContext") -> None:
        ...

    def on_step_complete(self, step_name: str, context: "PipelineContext") -> None:
        ...

    def on_step_error(self, step_name: str, context: "PipelineContext", error: Exception) -> None:
        ...


class LoggingObserver:
    def __init__(self, logger: Callable[[str], None]) -> None:
        self.logger = logger

    def on_step_start(self, step_name: str, context: "PipelineContext") -> None:
        self.logger(f"[pipeline] step start: {step_name}")

    def on_step_complete(self, step_name: str, context: "PipelineContext") -> None:
        self.logger(f"[pipeline] step complete: {step_name}")

    def on_step_error(self, step_name: str, context: "PipelineContext", error: Exception) -> None:
        self.logger(f"[pipeline] step error: {step_name} -> {error!r}")


@dataclass
class PipelineContext:
    topic: str
    base_dir: str
    reuse_existing_run: bool
    observer: Optional[PipelineObserver] = None
    data: dict = field(default_factory=dict)

    def set(self, key: str, value) -> None:
        self.data[key] = value

    def get(self, key: str, default=None):
        return self.data.get(key, default)


class PipelineStep(Protocol):
    name: str

    def set_next(self, step: "PipelineStep") -> "PipelineStep":
        ...

    def execute(self, context: PipelineContext) -> PipelineContext:
        ...


class BaseStep:
    def __init__(self, name: str):
        self.name = name
        self._next: Optional[PipelineStep] = None

    def set_next(self, step: PipelineStep) -> PipelineStep:
        self._next = step
        return step

    def execute(self, context: PipelineContext) -> PipelineContext:
        observer = context.observer
        try:
            if observer:
                observer.on_step_start(self.name, context)
            self.run(context)
            if observer:
                observer.on_step_complete(self.name, context)
        except Exception as exc:  # pragma: no cover
            if observer:
                observer.on_step_error(self.name, context, exc)
            raise

        if self._next:
            return self._next.execute(context)
        return context

    def run(self, context: PipelineContext) -> None:
        raise NotImplementedError


class PipelineBuilder:
    def __init__(self) -> None:
        self._steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep) -> "PipelineBuilder":
        if self._steps:
            self._steps[-1].set_next(step)
        self._steps.append(step)
        return self

    def build(self) -> PipelineStep:
        if not self._steps:
            raise RuntimeError("Pipeline has no steps configured.")
        return self._steps[0]
