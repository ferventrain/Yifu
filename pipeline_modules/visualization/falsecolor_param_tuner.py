"""Shim: falsecolor param tuner now lives under pipeline_modules.HE."""

from pipeline_modules.HE.falsecolor_param_tuner import *  # noqa: F403
from pipeline_modules.HE.falsecolor_param_tuner import main

if __name__ == "__main__":
    raise SystemExit(main())
