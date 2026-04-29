"""
Event-Level Baseline — 이벤트 단위 FAISS 분류.

GLIDE의 SentenceTransformer + FAISS 인덱스 재사용, 그룹 없이 개별 이벤트를 쿼리.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_FINAL_CODE = Path(__file__).resolve().parent.parent.parent.parent
if str(_FINAL_CODE) not in sys.path:
    sys.path.insert(0, str(_FINAL_CODE))

from experiments.baselines.common.adapter import BaselineAdapter, BaselinePrediction


_MEANINGFUL_EIDS = {1, 4688, 4104, 10, 11, 12, 13, 22, 23, 8, 4698, 7045}


def _event_to_text(obj: dict) -> str:
    """이벤트 하나를 FAISS 쿼리용 텍스트로 직렬화."""
    eid = obj.get("EventID") or obj.get("event_id")
    parts: list[str] = []
    if eid == 1 or eid == 4688:
        parts.append(f"Process created: {obj.get('Image') or obj.get('NewProcessName') or ''}")
        cl = obj.get("CommandLine") or ""
        if cl: parts.append(f"Command: {cl[:300]}")
        pi = obj.get("ParentImage") or obj.get("ParentProcessName") or ""
        if pi: parts.append(f"Parent: {pi}")
    elif eid == 4104:
        sb = obj.get("ScriptBlockText") or ""
        parts.append(f"PowerShell script block: {sb[:500]}")
    elif eid == 10:
        parts.append(f"Process access: {obj.get('SourceImage','')} -> {obj.get('TargetImage','')} (access={obj.get('GrantedAccess','')})")
    elif eid == 11:
        parts.append(f"File created: {obj.get('TargetFilename','')} by {obj.get('Image','')}")
    elif eid in (12, 13, 14):
        parts.append(f"Registry {obj.get('EventType','modified')}: {obj.get('TargetObject','')} = {obj.get('Details','')}")
    elif eid == 22:
        parts.append(f"DNS query: {obj.get('QueryName','')}")
    elif eid == 23:
        parts.append(f"File deleted: {obj.get('TargetFilename','')} by {obj.get('Image','')}")
    elif eid == 8:
        parts.append(f"CreateRemoteThread: {obj.get('SourceImage','')} -> {obj.get('TargetImage','')}")
    else:
        parts.append(f"EID {eid}: {obj.get('Message','')[:300]}")
    return ". ".join(parts)


class EventLevelAdapter(BaselineAdapter):
    name = "event_level"

    def __init__(self):
        from pipeline.mitre_mapper import get_embed_model, build_or_load_faiss_index, search_similar
        import config
        self._embed = get_embed_model()
        self._index, self._meta = build_or_load_faiss_index(
            str(config.MITRE_CSV_PATH), self._embed, cache_dir=config.CACHE_DIR,
        )
        self._search_similar = search_similar
        from pipeline.attack_chain import load_tactic_map
        self._tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))

    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        from pipeline.data_loader import load_and_normalize

        df = load_and_normalize(str(scenario_json_path))
        mask = df["EventID"].isin(_MEANINGFUL_EIDS)
        df = df[mask]

        tactic_seq: list[str] = []
        tech_seq:   list[str] = []
        per_group_topk: list[list[dict]] = []

        for _, row in df.iterrows():
            text = _event_to_text(row.to_dict())
            if not text.strip():
                continue
            similar, _ = self._search_similar(text, self._index, self._meta, self._embed, k=5)
            if not similar:
                continue
            top1 = similar[0]
            tid = top1["technique_id"]
            tech_seq.append(tid)
            tactics = self._tactic_map.get(tid) or self._tactic_map.get(tid.split(".")[0])
            if tactics:
                tactic_seq.append(tactics[0])
            per_group_topk.append(similar)

        return BaselinePrediction(
            scenario=scenario_json_path.stem,
            tactic_sequence=tactic_seq,
            technique_sequence=tech_seq,
            per_group_topk=per_group_topk,
            notes={"num_events_scored": len(tech_seq)},
        )
