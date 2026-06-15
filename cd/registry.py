"""Lightweight, file-based model registry (CD pillar).

A self-contained alternative to MLflow / Hugging Face Hub for this project:
tracks model versions, their metrics, and a deployment stage
(Staging -> Production -> Archived) in a JSON file. Promoting a version to
Production automatically archives the previous Production version.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

STAGES = ("Staging", "Production", "Archived")


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self._write({"models": []})

    # -- persistence -------------------------------------------------------
    def _read(self) -> dict:
        return json.loads(self.path.read_text())

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2))

    # -- operations --------------------------------------------------------
    def register(self, name: str, metrics: dict, artifact: str,
                 stage: str = "Staging") -> dict:
        """Register a new version of `name`. Version auto-increments."""
        if stage not in STAGES:
            raise ValueError(f"Invalid stage: {stage}")
        data = self._read()
        version = 1 + max([e["version"] for e in data["models"] if e["name"] == name],
                          default=0)
        entry = {
            "name": name,
            "version": version,
            "stage": stage,
            "metrics": metrics,
            "artifact": artifact,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        data["models"].append(entry)
        self._write(data)
        return entry

    def promote(self, name: str, version: int, stage: str = "Production") -> dict:
        """Move a version to a stage; archive the previous Production version."""
        if stage not in STAGES:
            raise ValueError(f"Invalid stage: {stage}")
        data = self._read()
        target = None
        for entry in data["models"]:
            if entry["name"] == name and entry["version"] == version:
                target = entry
            elif (stage == "Production" and entry["name"] == name
                  and entry["stage"] == "Production"):
                entry["stage"] = "Archived"
        if target is None:
            raise KeyError(f"{name} v{version} not found")
        target["stage"] = stage
        self._write(data)
        return target

    def get_production(self, name: str) -> dict | None:
        for entry in self._read()["models"]:
            if entry["name"] == name and entry["stage"] == "Production":
                return entry
        return None

    def list_versions(self, name: str) -> list:
        return [e for e in self._read()["models"] if e["name"] == name]
