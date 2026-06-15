"""Tests for the file-based model registry (CD pillar)."""
import pytest

from cd.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry.json")


def test_register_auto_increments_version(registry):
    v1 = registry.register("rul_model", {"mae_days": 6.8}, "Model/rul_model.pkl")
    v2 = registry.register("rul_model", {"mae_days": 6.5}, "Model/rul_model.pkl")
    assert v1["version"] == 1
    assert v2["version"] == 2
    assert v1["stage"] == "Staging"


def test_promote_sets_production_and_archives_previous(registry):
    registry.register("rul_model", {"mae_days": 6.8}, "a.pkl")
    registry.register("rul_model", {"mae_days": 6.5}, "b.pkl")
    registry.promote("rul_model", 1, "Production")
    registry.promote("rul_model", 2, "Production")

    prod = registry.get_production("rul_model")
    assert prod["version"] == 2
    v1 = next(e for e in registry.list_versions("rul_model") if e["version"] == 1)
    assert v1["stage"] == "Archived"


def test_get_production_none_when_only_staging(registry):
    registry.register("rul_model", {"mae_days": 6.8}, "a.pkl")
    assert registry.get_production("rul_model") is None


def test_promote_unknown_version_raises(registry):
    with pytest.raises(KeyError):
        registry.promote("rul_model", 99, "Production")


def test_invalid_stage_raises(registry):
    with pytest.raises(ValueError):
        registry.register("rul_model", {}, "a.pkl", stage="Live")
