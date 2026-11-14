from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar
import logging
import threading
import traceback
import time

TState = TypeVar("TState")

_fallback_logger = logging.getLogger("planproof_geometry.pipeline")


class StepTimeoutError(Exception):
    """Raised when a pipeline step exceeds its timeout."""


class Logger(Protocol):
    def __call__(self, msg: str) -> None: ...


@dataclass
class PipelineContext:
    """Context shared across pipeline steps."""
    workdir: str
    logger: Optional[Logger] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def log(self, msg: str) -> None:
        if self.logger:
            self.logger(msg)
        else:
            _fallback_logger.info(msg)


class Step(Protocol, Generic[TState]):
    """A single pipeline step."""
    name: str
    def run(self, ctx: PipelineContext, state: TState) -> TState: ...


@dataclass
class StepResult:
    name: str
    ok: bool
    started_at: float
    finished_at: float
    error: Optional[str] = None

@dataclass
class Artifact:
    name: str
    mime: str
    path: str  

@dataclass
class PipelineResult(Generic[TState]):
    ok: bool
    state: Optional[TState]
    steps: List[StepResult]
    result: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Artifact] = field(default_factory=list)
    error: Optional[str] = None

def _run_step_with_timeout(
    step: Step[TState],
    ctx: PipelineContext,
    state: TState,
    timeout: Optional[float],
) -> TState:
    """Execute a single step, optionally with a timeout.

    If *timeout* is ``None`` the step runs synchronously in the calling thread.
    Otherwise a worker thread is used and ``StepTimeoutError`` is raised when
    the deadline is exceeded.
    """
    if timeout is None:
        return step.run(ctx, state)

    result_holder: Dict[str, Any] = {}
    error_holder: Dict[str, Exception] = {}

    def _target():
        try:
            result_holder["state"] = step.run(ctx, state)
        except Exception as exc:
            error_holder["exc"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        step_name = getattr(step, "name", step.__class__.__name__)
        raise StepTimeoutError(
            f"Step '{step_name}' exceeded timeout of {timeout}s"
        )

    if "exc" in error_holder:
        raise error_holder["exc"]

    return result_holder["state"]


def run_pipeline(
    ctx: PipelineContext,
    steps: List[Step[TState]],
    initial_state: TState,
    *,
    default_step_timeout: Optional[float] = None,
) -> PipelineResult[TState]:
    """Runs steps sequentially with consistent logging + error handling + timing.

    Steps with ``soft_fail = True`` will log errors but not abort the pipeline.

    Each step may declare a ``timeout`` attribute (seconds).  If set, the step
    is executed in a worker thread and ``StepTimeoutError`` is raised when the
    deadline is exceeded.  ``default_step_timeout`` applies to steps that do
    not declare their own timeout.
    """
    state: TState = initial_state
    results: List[StepResult] = []

    for step in steps:
        started = time.time()
        step_name = getattr(step, "name", step.__class__.__name__)
        soft_fail = getattr(step, "soft_fail", False)
        step_timeout = getattr(step, "timeout", default_step_timeout)
        ctx.log(f"▶ {step_name}")
        try:
            state = _run_step_with_timeout(step, ctx, state, step_timeout)
            results.append(
                StepResult(
                    name=step_name,
                    ok=True,
                    started_at=started,
                    finished_at=time.time(),
                )
            )
        except Exception as e:
            tb = traceback.format_exc()
            err_type = type(e).__name__
            msg = f"[{step_name}] {err_type}: {e}\n{tb}"
            if soft_fail:
                ctx.log(f"⚠ Nicht-kritischer Fehler in '{step_name}': {err_type}: {e} (wird uebersprungen)")
                results.append(
                    StepResult(
                        name=step_name,
                        ok=False,
                        started_at=started,
                        finished_at=time.time(),
                        error=f"[soft_fail] {err_type}: {e}",
                    )
                )
                # Continue with unchanged state
            else:
                ctx.log(f"✖ Fehler in Schritt '{step_name}': {err_type}: {e}")
                results.append(
                    StepResult(
                        name=step_name,
                        ok=False,
                        started_at=started,
                        finished_at=time.time(),
                        error=f"[{step_name}] {err_type}: {e}",
                    )
                )
                return PipelineResult(ok=False, state=None, steps=results, error=msg)

    return PipelineResult(ok=True, state=state, steps=results, error=None)
