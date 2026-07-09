import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_cfos_3d_mlflow import (  # noqa: E402
    accumulate_foreground_counts,
    compute_metrics,
    metrics_from_counts,
)


def test_metrics_from_counts_empty_prediction_on_empty_target():
    stats = metrics_from_counts(tp=0.0, fp=0.0, fn=0.0)
    assert stats["global_precision"] == 1.0
    assert stats["global_recall"] == 1.0
    assert stats["global_dice"] == 1.0


def test_metrics_from_counts_false_positives_reduce_global_precision():
    stats = metrics_from_counts(tp=0.0, fp=100.0, fn=0.0)
    assert stats["global_precision"] == 0.0
    assert stats["global_recall"] == 1.0


def test_macro_dice_can_be_zero_while_global_precision_is_perfect():
    logits = torch.zeros((2, 2, 4, 4, 4))
    logits[:, 0, ...] = 10.0
    target = torch.zeros((2, 4, 4, 4), dtype=torch.long)

    macro = compute_metrics(logits, target, num_classes=2)
    assert macro["dice"] == 1.0

    pred = torch.argmax(logits, dim=1)
    tp, fp, fn = accumulate_foreground_counts(pred, target, num_classes=2)
    global_stats = metrics_from_counts(tp, fp, fn)
    assert global_stats["global_precision"] == 1.0
    assert global_stats["global_dice"] == 1.0


def test_global_metrics_penalize_scattered_false_positives():
    logits = torch.full((1, 2, 4, 4, 4), -10.0)
    logits[:, 0, ...] = 10.0
    logits[:, 1, 0, 0, 0] = 20.0
    target = torch.zeros((1, 4, 4, 4), dtype=torch.long)

    pred = torch.argmax(logits, dim=1)
    tp, fp, fn = accumulate_foreground_counts(pred, target, num_classes=2)
    global_stats = metrics_from_counts(tp, fp, fn)
    assert tp == 0.0
    assert fp == 1.0
    assert global_stats["global_precision"] == 0.0
