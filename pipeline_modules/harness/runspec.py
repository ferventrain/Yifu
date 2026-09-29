"""RunSpec: the only shape of work the harness accepts.

A RunSpec describes ONE pipeline run. It is deliberately closed: ``pipeline``
must be a known pipeline id (never a shell command), and the run itself is
always ``main.py --config <config> --sample_dir <sample>`` plus a small list
of pre-declared extra args. Agents cannot smuggle arbitrary commands through
the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline_modules.harness.progress import PIPELINE_MODE_BRAIN, PIPELINE_MODE_SPINAL_CORD, pipeline_mode
from pipeline_modules.utils.errors import ErrorCode, PipelineError

KNOWN_PIPELINES = (PIPELINE_MODE_BRAIN, PIPELINE_MODE_SPINAL_CORD)

#: Extra main.py flags the harness is allowed to forward. Anything else is a
#: configuration smell and must be rejected at submit time, not mid-run.
ALLOWED_EXTRA_FLAGS = {"--skip_registration", "--only_tubule", "--only_region_analysis"}


@dataclass(frozen=True)
class RunSpec:
    sample_dir: Path
    config_path: Path
    pipeline: str = PIPELINE_MODE_BRAIN
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_args(
        cls,
        sample_dir: str | Path,
        config_path: str | Path | None = None,
        *,
        extra_args: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "RunSpec":
        sample = Path(sample_dir).expanduser().resolve()
        config_file = Path(config_path).expanduser().resolve() if config_path else sample / "config.json"
        if not sample.is_dir():
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "sample_dir does not exist",
                context={"sample_dir": str(sample)},
            )
        if not config_file.is_file():
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "config.json not found",
                context={"config_path": str(config_file), "hint": "pass --config explicitly"},
            )
        extras = tuple(str(item) for item in (extra_args or []))
        bad = [item for item in extras if item not in ALLOWED_EXTRA_FLAGS]
        if bad:
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "unsupported extra argument; the run command only forwards pre-declared flags",
                context={"rejected": bad, "allowed": sorted(ALLOWED_EXTRA_FLAGS)},
            )
        if config is None:
            from pipeline_modules.harness.results import load_config

            try:
                config = load_config(config_file)
            except (OSError, ValueError) as exc:
                raise PipelineError(
                    ErrorCode.CONFIG_INVALID,
                    "config.json is not valid JSON",
                    context={"config_path": str(config_file), "error": repr(exc)},
                ) from exc
        mode = pipeline_mode(config)
        return cls(sample_dir=sample, config_path=config_file, pipeline=mode, extra_args=extras)

    def command(self, python_exe: str, main_py: Path, progress_file: Path) -> list[str]:
        """The one standard pipeline command this spec maps to."""
        command = [
            python_exe,
            str(main_py),
            "--config",
            str(self.config_path),
            "--sample_dir",
            str(self.sample_dir),
            "--progress_file",
            str(progress_file),
        ]
        command.extend(self.extra_args)
        return command
