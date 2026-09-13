---
name: yifu-test
description: Placeholder for the Yifu harness test mode that will sweep algorithm parameters at many volume locations with visual results. Use when the user asks to test or tune pipeline parameters, compare settings, or inspect multi-ROI previews.
---

# Yifu test mode (not implemented in v1)

Test mode is a reserved harness tab. Do not build a parameter-sweep UI or multi-ROI visualizer unless the user asks to implement that feature.

Until then:

- Point the user at the **测试** tab in `python -m apps.pipeline_harness --host 127.0.0.1 --port 8766`.
- For ad-hoc tuning, use existing module CLIs (for example threshold `--test`, falsecolor param tuner, SpinalJ wizard) rather than inventing a new harness runner.
- Planned later: sample many XYZ crops, run one algorithm under several parameter sets, and show side-by-side previews.
