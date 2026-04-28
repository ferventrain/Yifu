"""Structured error model shared across pipeline modules.

Every pipeline module should raise ``PipelineError`` (or a subclass) with a
well-defined ``ErrorCode`` instead of calling ``sys.exit`` or letting raw
``ValueError`` / ``FileNotFoundError`` escape the CLI layer. This lets agents
(and humans) distinguish between user errors, missing inputs, resource
exhaustion, and internal bugs without scraping ``stderr``.

CLI entrypoints should catch ``PipelineError``, print a JSON error body to
stderr, and exit with the error's mapped exit code.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ErrorCode(str, Enum):
    """Enumeration of structured error codes.

    Values are short uppercase slugs intended for machine-readable consumption
    (logs, JSON error bodies, capability manifests).
    """

    # --- User / configuration errors (exit code 2) ---
    CONFIG_INVALID = "CONFIG_INVALID"
    ARGUMENT_INVALID = "ARGUMENT_INVALID"

    # --- Input / environment errors (exit code 3) ---
    INPUT_NOT_FOUND = "INPUT_NOT_FOUND"
    INPUT_FORMAT_INVALID = "INPUT_FORMAT_INVALID"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"

    # --- Runtime / resource errors (exit code 4) ---
    CUDA_OOM = "CUDA_OOM"
    CONVERGENCE_FAILED = "CONVERGENCE_FAILED"
    EMPTY_RESULT = "EMPTY_RESULT"

    # --- Internal / unexpected (exit code 5) ---
    INTERNAL_ERROR = "INTERNAL_ERROR"


_EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.CONFIG_INVALID: 2,
    ErrorCode.ARGUMENT_INVALID: 2,
    ErrorCode.INPUT_NOT_FOUND: 3,
    ErrorCode.INPUT_FORMAT_INVALID: 3,
    ErrorCode.DEPENDENCY_MISSING: 3,
    ErrorCode.CUDA_OOM: 4,
    ErrorCode.CONVERGENCE_FAILED: 4,
    ErrorCode.EMPTY_RESULT: 4,
    ErrorCode.INTERNAL_ERROR: 5,
}


class PipelineError(Exception):
    """Base class for all structured pipeline errors.

    Parameters
    ----------
    code
        A value from :class:`ErrorCode` indicating the error category.
    message
        Human-readable description. Keep it short; attach structured details
        via ``context`` rather than inlining them here.
    context
        Optional mapping of machine-readable context (paths, parameter
        values, upstream exception repr, etc.). Must be JSON-serializable.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = ErrorCode(code)
        self.message = str(message)
        self.context: dict[str, Any] = dict(context) if context else {}

    @property
    def exit_code(self) -> int:
        return _EXIT_CODES.get(self.code, 1)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation suitable for CLI stderr output."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "context": self.context,
            }
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"PipelineError(code={self.code.value!r}, message={self.message!r})"
