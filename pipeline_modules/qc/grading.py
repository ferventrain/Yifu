from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Direction = Literal["lower_is_better", "higher_is_better"]
Verdict = Literal["pass", "warn", "fail"]


@dataclass
class ThresholdRule:
    metric: str
    direction: Direction
    pass_max: float | None = None
    pass_min: float | None = None
    warn_max: float | None = None
    warn_min: float | None = None
    category: str = ""
    description: str = ""


DEFAULT_THRESHOLD_RULES: list[ThresholdRule] = [
    ThresholdRule(
        metric="global_exposure_dynamic_range.dark_pixel_ratio",
        direction="lower_is_better",
        pass_max=0.05,
        warn_max=0.15,
        category="exposure_dynamic_range",
        description="Fraction of pixels below dark threshold (default <100).",
    ),
    ThresholdRule(
        metric="global_exposure_dynamic_range.saturated_pixel_ratio",
        direction="lower_is_better",
        pass_max=0.001,
        warn_max=0.01,
        category="exposure_dynamic_range",
        description="Fraction of saturated/near-max pixels.",
    ),
    ThresholdRule(
        metric="slice_aggregate.brightness_uniformity.tile_median_cv.median",
        direction="lower_is_better",
        pass_max=0.15,
        warn_max=0.30,
        category="brightness_uniformity",
        description="Median tile-brightness coefficient of variation across sampled slices.",
    ),
    ThresholdRule(
        metric="slice_aggregate.stripe_artifacts.slab_stripe_score.median",
        direction="lower_is_better",
        pass_max=0.08,
        warn_max=0.20,
        category="stripe_artifacts",
        description="Combined row/column/slab stripe score.",
    ),
    ThresholdRule(
        metric="slice_aggregate.diffuse_signal.diffuse_noise_score.median",
        direction="lower_is_better",
        pass_max=0.20,
        warn_max=0.40,
        category="diffuse_signal",
        description="Low-contrast elevated area likely to be diffuse noise rather than structured signal.",
    ),
    ThresholdRule(
        metric="slice_aggregate.diffuse_signal.noise_to_signal_area_ratio.median",
        direction="lower_is_better",
        pass_max=0.50,
        warn_max=1.50,
        category="diffuse_signal",
        description="Ratio of diffuse-noise tile area to structured-signal tile area.",
    ),
    ThresholdRule(
        metric="slice_aggregate.diffuse_signal.large_diffuse_component_fraction.median",
        direction="lower_is_better",
        pass_max=0.10,
        warn_max=0.25,
        category="diffuse_signal",
        description="Largest connected diffuse-noise tile component as fraction of tile grid.",
    ),
    ThresholdRule(
        metric="slice_aggregate.contrast.candidate_cnr.median",
        direction="higher_is_better",
        pass_min=3.0,
        warn_min=1.5,
        category="contrast",
        description="Robust Otsu-based candidate contrast-to-noise ratio.",
    ),
    ThresholdRule(
        metric="slice_aggregate.focus.laplacian_variance.median",
        direction="higher_is_better",
        pass_min=50.0,
        warn_min=20.0,
        category="focus",
        description="Median Laplacian variance across sampled slices (resolution dependent).",
    ),
]


def _lookup_metric(flat_metrics: dict[str, Any], metric_path: str) -> float | None:
    if metric_path in flat_metrics:
        value = flat_metrics[metric_path]
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    parts = metric_path.split(".")
    current: Any = flat_metrics
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _grade_value(value: float, rule: ThresholdRule) -> Verdict:
    if rule.direction == "lower_is_better":
        if rule.pass_max is not None and value <= rule.pass_max:
            return "pass"
        if rule.warn_max is not None and value <= rule.warn_max:
            return "warn"
        return "fail"
    if rule.pass_min is not None and value >= rule.pass_min:
        return "pass"
    if rule.warn_min is not None and value >= rule.warn_min:
        return "warn"
    return "fail"


def _overall_verdict(verdicts: list[Verdict]) -> Verdict:
    if any(v == "fail" for v in verdicts):
        return "fail"
    if any(v == "warn" for v in verdicts):
        return "warn"
    return "pass"


def flatten_qc_metrics_for_grading(results: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    global_exposure = results.get("global_exposure_dynamic_range")
    if isinstance(global_exposure, dict):
        for key, value in global_exposure.items():
            flat[f"global_exposure_dynamic_range.{key}"] = value
    slice_aggregate = results.get("slice_aggregate")
    if isinstance(slice_aggregate, dict):
        flat.update(slice_aggregate)
    projection = results.get("projection_metrics")
    if isinstance(projection, dict):
        metrics_flat = projection.get("metrics_flat")
        if isinstance(metrics_flat, dict):
            flat.update(metrics_flat)
    return flat


def grade_qc_results(
    results: dict[str, Any],
    *,
    rules: list[ThresholdRule] | None = None,
) -> dict[str, Any]:
    rule_list = rules or DEFAULT_THRESHOLD_RULES
    flat_metrics = flatten_qc_metrics_for_grading(results)
    rule_results: list[dict[str, Any]] = []
    verdicts: list[Verdict] = []

    for rule in rule_list:
        value = _lookup_metric(flat_metrics, rule.metric)
        if value is None:
            entry = {
                "metric": rule.metric,
                "category": rule.category,
                "description": rule.description,
                "direction": rule.direction,
                "value": None,
                "verdict": "warn",
                "reason": "metric_missing",
            }
            rule_results.append(entry)
            verdicts.append("warn")
            continue

        verdict = _grade_value(value, rule)
        entry = {
            "metric": rule.metric,
            "category": rule.category,
            "description": rule.description,
            "direction": rule.direction,
            "value": value,
            "verdict": verdict,
            "thresholds": {
                "pass_max": rule.pass_max,
                "pass_min": rule.pass_min,
                "warn_max": rule.warn_max,
                "warn_min": rule.warn_min,
            },
        }
        rule_results.append(entry)
        verdicts.append(verdict)

    by_category: dict[str, list[Verdict]] = {}
    for entry in rule_results:
        category = str(entry.get("category") or "other")
        by_category.setdefault(category, []).append(entry["verdict"])

    category_summary = {
        category: _overall_verdict(items) for category, items in by_category.items()
    }
    overall = _overall_verdict(verdicts)

    return {
        "overall_verdict": overall,
        "category_verdicts": category_summary,
        "rules": rule_results,
        "notes": (
            "Starter thresholds for review only. Tune pass/warn cutoffs on a reference "
            "sample cohort before production gating."
        ),
    }


def load_threshold_rules(path: str | Path | None) -> list[ThresholdRule]:
    if path is None:
        return list(DEFAULT_THRESHOLD_RULES)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rules_payload = payload.get("rules", payload)
    rules: list[ThresholdRule] = []
    for item in rules_payload:
        rules.append(
            ThresholdRule(
                metric=str(item["metric"]),
                direction=item.get("direction", "lower_is_better"),
                pass_max=item.get("pass_max"),
                pass_min=item.get("pass_min"),
                warn_max=item.get("warn_max"),
                warn_min=item.get("warn_min"),
                category=str(item.get("category", "")),
                description=str(item.get("description", "")),
            )
        )
    return rules


def export_default_thresholds(path: str | Path) -> Path:
    payload = {
        "schema_version": "1",
        "notes": "Edit pass/warn cutoffs after reviewing QC outputs on reference samples.",
        "rules": [
            {
                "metric": rule.metric,
                "direction": rule.direction,
                "pass_max": rule.pass_max,
                "pass_min": rule.pass_min,
                "warn_max": rule.warn_max,
                "warn_min": rule.warn_min,
                "category": rule.category,
                "description": rule.description,
            }
            for rule in DEFAULT_THRESHOLD_RULES
        ],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
