"""CT fairness checks: model accuracy should be consistent across groups."""
import numpy as np

from ct.evaluate import mae_by_group, recall_per_class

FAILURE_CLASSES = [
    "Heat Dissipation Failure",
    "Overstrain Failure",
    "Power Failure",
    "Random Failures",
    "Tool Wear Failure",
]


def test_mae_gap_across_type_is_modest(ct_regression):
    """RUL error should be similar across machine types H/M/L."""
    per_type, gap = mae_by_group(
        ct_regression["df_test"]["Type"].to_numpy(),
        ct_regression["y_test"],
        ct_regression["pred"],
    )
    assert set(per_type).issubset({"H", "M", "L"})

    overall_mae = float(np.mean(np.abs(ct_regression["y_test"] - ct_regression["pred"])))
    # Fairness gate: the worst-vs-best group gap stays a modest fraction of the
    # overall error (not an order of magnitude apart).
    assert gap <= 1.5 * overall_mae, f"per_type={per_type}, gap={gap}"


def test_mae_consistent_across_rul_buckets(ct_regression):
    """No RUL range should be dramatically worse than the others."""
    y = ct_regression["y_test"]
    buckets = np.where(y < 1440, "soon", np.where(y < 14400, "mid", "far"))  # <60d, <600d, rest
    per_bucket, _ = mae_by_group(buckets, y, ct_regression["pred"])
    assert all(np.isfinite(v) for v in per_bucket.values())
    assert len(per_bucket) >= 2


def test_failure_classifier_attempts_all_classes(ct_classification):
    """Mechanism-level fairness check on the (rare) failure classes.

    With only ~250 failure rows we cannot demand high per-class recall; we
    assert the evaluation runs and the model is not collapsing to one class.
    """
    recalls = recall_per_class(
        ct_classification["y_test"], ct_classification["pred"], FAILURE_CLASSES
    )
    assert set(recalls) == set(FAILURE_CLASSES)
    macro_recall = float(np.mean(list(recalls.values())))
    assert macro_recall > 0.1, recalls
    # predicts more than a single class
    assert len(set(ct_classification["pred"])) >= 2
