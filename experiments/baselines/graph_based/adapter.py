"""

"""
from __future__ import annotations

import sys
from pathlib import Path

_FINAL_CODE = Path(__file__).resolve().parent.parent.parent.parent
if str(_FINAL_CODE) not in sys.path:
    sys.path.insert(0, str(_FINAL_CODE))

from experiments.baselines.common.adapter import BaselineAdapter, BaselinePrediction


class OCRAptAdapter(BaselineAdapter):
    """OCR-APT ...repo...attack chain ..."""
    name = "graph_ocr_apt"

    def __init__(self, repo_path: Path | None = None):
        self.repo_path = repo_path

    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        # TODO:
        raise NotImplementedError("OCR-APT ...repo ...")


class MarlinAdapter(BaselineAdapter):
    """MARLIN ...repo..."""
    name = "graph_marlin"

    def __init__(self, repo_path: Path | None = None):
        self.repo_path = repo_path

    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        raise NotImplementedError("MARLIN ...repo ...")
