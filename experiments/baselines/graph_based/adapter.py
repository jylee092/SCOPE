"""
Graph-based baseline 어댑터 스텁 — OCR-APT, MARLIN.

실제 구현은 외부 repo 통합 후 완성.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FINAL_CODE = Path(__file__).resolve().parent.parent.parent.parent
if str(_FINAL_CODE) not in sys.path:
    sys.path.insert(0, str(_FINAL_CODE))

from experiments.baselines.common.adapter import BaselineAdapter, BaselinePrediction


class OCRAptAdapter(BaselineAdapter):
    """OCR-APT 외부 repo를 호출하여 attack chain 재구성."""
    name = "graph_ocr_apt"

    def __init__(self, repo_path: Path | None = None):
        self.repo_path = repo_path
        # TODO: 외부 repo 경로 검증, 의존성 로드

    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        # TODO:
        # 1. scenario_json_path → provenance graph 변환 (sysmon_to_provenance)
        # 2. OCR-APT 호출 (subprocess or Python API)
        # 3. 출력 subgraph → tactic/technique 시퀀스 추출
        raise NotImplementedError("OCR-APT 외부 repo 통합 후 구현")


class MarlinAdapter(BaselineAdapter):
    """MARLIN 외부 repo를 호출."""
    name = "graph_marlin"

    def __init__(self, repo_path: Path | None = None):
        self.repo_path = repo_path

    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        raise NotImplementedError("MARLIN 외부 repo 통합 후 구현")
