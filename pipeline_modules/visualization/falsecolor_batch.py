"""Shim: falsecolor batch now lives under pipeline_modules.HE."""

from pipeline_modules.HE.falsecolor_batch import *  # noqa: F403
from pipeline_modules.HE.falsecolor_batch import main

if __name__ == "__main__":
    raise SystemExit(main())
