"""Batch-oriented helpers for reading and writing large TIFF stacks."""

from __future__ import annotations

import os
import sys
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import numpy as np
import tifffile

JobT = TypeVar("JobT")


def iter_batch_ranges(depth: int, batch_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, depth, batch_size):
        yield start, min(start + batch_size, depth)


def stack_tiff_paths(paths: list[Path]) -> np.ndarray:
    return np.stack([tifffile.imread(str(path)) for path in paths])


def normalize_tiff_compression(compression: str | None) -> str | None:
    if compression is None:
        return None
    text = str(compression).strip().lower()
    if text in ("", "none", "off", "false", "uncompressed", "no"):
        return None
    return str(compression)


def resolve_stack_workers(workers: int, *, default_cap: int = 4) -> int:
    if workers < 0:
        raise ValueError(f"workers must be >= 0, got {workers}")
    if workers == 0:
        worker_count = max(1, min(default_cap, (os.cpu_count() or 4) // 4))
    else:
        worker_count = int(workers)
    if os.name == "nt" and worker_count > 61:
        worker_count = 61
    return worker_count


def resolve_slice_batch(slice_batch: int | None, *, default: int = 16) -> int:
    if slice_batch is None or slice_batch <= 0:
        return default
    return max(1, int(slice_batch))


def write_tiff_stack_batch(
    slices: np.ndarray,
    output_paths: list[Path],
    *,
    compression: str | None = None,
) -> None:
    comp = normalize_tiff_compression(compression)
    for offset, output_path in enumerate(output_paths):
        tifffile.imwrite(str(output_path), slices[offset], compression=comp)


def run_bounded_batches(
    jobs: list[JobT],
    worker_fn: Callable[[JobT], int],
    *,
    worker_count: int,
    progress_total: int,
    desc: str,
    unit: str = "slice",
    executor_cls: type[Executor] = ThreadPoolExecutor,
    progress_file: Any | None = None,
) -> int:
    """Run batch jobs with bounded in-flight tasks. worker_fn returns completed unit count."""
    if not jobs:
        return 0

    from tqdm import tqdm

    completed_units = 0
    if worker_count <= 1:
        iterator: Any = jobs
        if progress_file is None:
            progress_file = sys.stderr
        if hasattr(progress_file, "isatty") and progress_file.isatty():
            iterator = tqdm(jobs, desc=desc, unit="batch", leave=False, file=progress_file)
        for job in iterator:
            completed_units += int(worker_fn(job))
        return completed_units

    max_in_flight = max(1, min(worker_count + 1, len(jobs)))
    pending: dict[Future[int], None] = {}
    job_iter = iter(jobs)
    with executor_cls(max_workers=worker_count) as pool:
        def submit_next() -> None:
            try:
                pending[pool.submit(worker_fn, next(job_iter))] = None
            except StopIteration:
                return

        for _ in range(max_in_flight):
            submit_next()

        if progress_file is None:
            progress_file = sys.stderr
        progress = tqdm(total=progress_total, desc=desc, unit=unit, leave=False, file=progress_file)
        try:
            while pending:
                for future in as_completed(list(pending)):
                    pending.pop(future)
                    batch_units = int(future.result())
                    completed_units += batch_units
                    progress.update(batch_units)
                    submit_next()
                    break
        finally:
            progress.close()
    return completed_units
