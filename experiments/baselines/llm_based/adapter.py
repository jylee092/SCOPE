"""

"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_FINAL_CODE = Path(__file__).resolve().parent.parent.parent.parent
if str(_FINAL_CODE) not in sys.path:
    sys.path.insert(0, str(_FINAL_CODE))

from experiments.baselines.common.adapter import BaselineAdapter, BaselinePrediction


PROMPT_TEMPLATE = """You are a cybersecurity analyst reviewing Windows Sysmon/Security logs.
Given the chronologically ordered event summaries below, reconstruct the MITRE ATT&CK attack chain.

# Events (chronological, truncated if long)
{events}

# Task
Output a JSON object with these fields only:
- "tactic_sequence": list of MITRE tactic names in execution order (e.g., "Execution", "Persistence")
- "technique_sequence": list of MITRE technique IDs in execution order (e.g., "T1059.001", "T1547.001")
- "reasoning": one sentence summary

Respond with JSON only, no commentary."""


_MAX_EVENTS = 300
_MEANINGFUL_EIDS = {1, 4688, 4104, 10, 11, 12, 13, 22, 23, 8, 4698, 7045}


def _summarize_event(row: dict) -> str:
    eid = row.get("EventID") or row.get("event_id")
    t = row.get("@timestamp") or row.get("TimeCreated") or ""
    ts = str(t)[:19]
    if eid == 1 or eid == 4688:
        img = row.get("Image") or row.get("NewProcessName") or "?"
        cl = (row.get("CommandLine") or "")[:150]
        return f"{ts} EID{eid} ProcessCreate {img} {cl}"
    if eid == 4104:
        sb = (row.get("ScriptBlockText") or "")[:200]
        return f"{ts} EID4104 PSScriptBlock {sb}"
    if eid == 10:
        return f"{ts} EID10 ProcessAccess {row.get('SourceImage','?')} -> {row.get('TargetImage','?')}"
    if eid == 11:
        return f"{ts} EID11 FileCreate {row.get('TargetFilename','')} by {row.get('Image','?')}"
    if eid in (12, 13, 14):
        return f"{ts} EID{eid} Registry {row.get('TargetObject','')} = {row.get('Details','')}"
    if eid == 22:
        return f"{ts} EID22 DnsQuery {row.get('QueryName','')}"
    if eid == 23:
        return f"{ts} EID23 FileDelete {row.get('TargetFilename','')}"
    return f"{ts} EID{eid}"


class ShieldLikeAdapter(BaselineAdapter):
    """SHIELD ...: ...Gemini → attack chain JSON."""
    name = "llm_shield"

    def __init__(self, model_name: str = None):
        import config
        import google.generativeai as genai
        self._api_key = config.GEMINI_API_KEY
        self._model_name = model_name or config.GEMINI_MODEL
        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(self._model_name)
        self._genai = genai

    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        from pipeline.data_loader import load_and_normalize

        df = load_and_normalize(str(scenario_json_path))
        if "EventID" in df.columns:
            df = df[df["EventID"].isin(_MEANINGFUL_EIDS)]
        if len(df) > _MAX_EVENTS:
            df = df.iloc[::len(df)//_MAX_EVENTS].head(_MAX_EVENTS)

        event_lines = [_summarize_event(r.to_dict()) for _, r in df.iterrows()]
        prompt = PROMPT_TEMPLATE.format(events="\n".join(event_lines))

        response = self._model.generate_content(
            prompt,
            generation_config=self._genai.types.GenerationConfig(
                temperature=0.2, max_output_tokens=800,
            ),
        )
        text = response.text.strip()

        m = re.search(r"\{[\s\S]*\}", text)
        parsed = {"tactic_sequence": [], "technique_sequence": [], "reasoning": ""}
        if m:
            try:
                parsed.update(json.loads(m.group()))
            except json.JSONDecodeError:
                pass

        return BaselinePrediction(
            scenario=scenario_json_path.stem,
            tactic_sequence=parsed.get("tactic_sequence", []),
            technique_sequence=parsed.get("technique_sequence", []),
            notes={
                "model": self._model_name,
                "events_given": len(event_lines),
                "raw_response": text[:1000],
                "reasoning": parsed.get("reasoning", ""),
            },
        )
