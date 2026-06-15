"""CT robustness checks: predictions should be stable under small input noise
and well-defined at the edges of the valid range."""
import numpy as np

from Preprocessing_pipeline import PreprocessingPipeline
from ct.evaluate import prediction_stability


def test_predictions_stable_under_small_noise(ct_regression):
    rel_change = prediction_stability(
        ct_regression["model"], ct_regression["X_test"], noise_scale=0.01, seed=0
    )
    # A 1% sensor perturbation should move predictions by only a few percent.
    assert rel_change < 0.15, f"relative change too high: {rel_change:.3f}"


def test_larger_noise_changes_more_than_small_noise(ct_regression):
    small = prediction_stability(ct_regression["model"], ct_regression["X_test"], 0.01, seed=1)
    large = prediction_stability(ct_regression["model"], ct_regression["X_test"], 0.05, seed=1)
    assert large >= small  # sanity: more noise -> at least as much movement


def test_extreme_valid_inputs_are_finite():
    """Boundary inputs at the edges of the valid range must not break features."""
    pipe = PreprocessingPipeline()
    for machine_type in ("H", "M", "L"):
        extreme = {
            "Type": machine_type,
            "Air_temperature": 400.0,
            "Process_temperature": 400.0,
            "Rotational_speed": 3000,
            "Torque": 100.0,
            "Tool_wear": 300,
        }
        feats = pipe.transform_single(extreme)
        assert feats.shape == (1, 18)
        assert np.all(np.isfinite(feats))


def test_optional_feature_defaults_are_safe(pipeline):
    """Omitting the optional/stateful features still yields a valid vector."""
    minimal = {
        "Type": "M", "Air_temperature": 300.0, "Process_temperature": 310.0,
        "Rotational_speed": 1500, "Torque": 40.0, "Tool_wear": 100,
    }
    feats = pipeline.transform_single(minimal)
    assert feats.shape == (1, 18)
    assert np.all(np.isfinite(feats))
