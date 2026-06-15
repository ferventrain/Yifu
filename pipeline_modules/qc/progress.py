from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, TypeVar


T = TypeVar("T")


@dataclass
class StepTiming:
    name: str
    label: str
    seconds: float = 0.0


@dataclass
class QcProgressTracker:
    enabled: bool = True
    bar_width: int = 40
    steps: list[StepTiming] = field(default_factory=list)

    @contextmanager
    def step(self, name: str, label: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.steps.append(
                StepTiming(
                    name=name,
                    label=label,
                    seconds=max(time.perf_counter() - started, 0.0),
                )
            )

    def iter(
        self,
        iterable: Any,
        *,
        desc: str,
        total: int | None = None,
        unit: str = "it",
    ) -> Iterator[T]:
        if not self.enabled:
            yield from iterable
            return
        try:
            from tqdm import tqdm
        except ModuleNotFoundError:
            yield from iterable
            return
        yield from tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            file=sys.stderr,
            dynamic_ncols=True,
            leave=False,
        )

    def total_seconds(self) -> float:
        return float(sum(step.seconds for step in self.steps))

    def to_dict(self) -> dict[str, Any]:
        total = self.total_seconds()
        rows = []
        for step in self.steps:
            fraction = step.seconds / total if total > 0 else 0.0
            rows.append(
                {
                    "name": step.name,
                    "label": step.label,
                    "seconds": round(step.seconds, 3),
                    "fraction": round(fraction, 4),
                    "percent": round(fraction * 100.0, 1),
                }
            )
        return {
            "total_seconds": round(total, 3),
            "steps": rows,
            "bar_chart": self.render_bars(),
        }

    def render_bars(self, *, total_seconds: float | None = None) -> str:
        total = total_seconds if total_seconds is not None else self.total_seconds()
        if total <= 0 or not self.steps:
            return "QC timing breakdown: no timed steps recorded."

        label_width = max(len(step.label) for step in self.steps)
        lines = [f"QC timing breakdown (total {total:.1f}s):"]
        for step in self.steps:
            fraction = step.seconds / total if total > 0 else 0.0
            filled = int(round(fraction * self.bar_width))
            filled = min(max(filled, 0), self.bar_width)
            bar = "█" * filled + "░" * (self.bar_width - filled)
            lines.append(
                f"{step.label:<{label_width}}  {bar}  {fraction * 100:5.1f}%  {step.seconds:6.1f}s"
            )
        return "\n".join(lines)

    def print_summary(self, *, total_seconds: float | None = None, file: Any = None) -> None:
        if not self.enabled or not self.steps:
            return
        handle = file if file is not None else sys.stderr
        print(self.render_bars(total_seconds=total_seconds), file=handle)


def count_zarr_blocks(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    total = 1
    for dim, chunk in zip(shape, chunks, strict=True):
        chunk_size = max(int(chunk), 1)
        total *= max((int(dim) + chunk_size - 1) // chunk_size, 1)
    return int(total)
