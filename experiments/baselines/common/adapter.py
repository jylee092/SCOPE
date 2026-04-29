"""

"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BaselinePrediction:
    """Baseline..."""
    scenario: str
    tactic_sequence: list[str] = field(default_factory=list)
    technique_sequence: list[str] = field(default_factory=list)
    per_group_topk: list[list[dict]] = field(default_factory=list)  # optional
    notes: dict = field(default_factory=dict)  # latency, token count, etc.


class BaselineAdapter:
    """...baseline..."""
    name: str = "base"

    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        """...JSON..."""
        raise NotImplementedError

    def save_result(self, pred: BaselinePrediction, out_dir: Path) -> Path:
        """...result.json..."""
        import json
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "scenario": pred.scenario,
                "tactic_sequence": pred.tactic_sequence,
                "technique_sequence": pred.technique_sequence,
                "per_group_topk": pred.per_group_topk,
                "notes": pred.notes,
            }, f, ensure_ascii=False, indent=2)
        return out_path


def load_prediction(path: Path) -> BaselinePrediction:
    """...result.json → BaselinePrediction."""
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return BaselinePrediction(
        scenario=data["scenario"],
        tactic_sequence=data.get("tactic_sequence", []),
        technique_sequence=data.get("technique_sequence", []),
        per_group_topk=data.get("per_group_topk", []),
        notes=data.get("notes", {}),
    )
