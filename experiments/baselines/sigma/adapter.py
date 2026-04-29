"""
Sigma baseline adapter — runs SigmaHQ Windows rules over a Mordor scenario
and emits a BaselinePrediction compatible with the common evaluator.

Per-alert representation:
    {
        "ts": <ISO timestamp>,
        "event_index": <int>,      # row index in the normalized DataFrame
        "event_id": <int>,
        "topk_tids": [...],        # priority-ordered, deduped, ≤ K
        "topk_tactics": [...],     # parallel to topk_tids
        "top_rule": {              # highest-priority matching rule for the event
            "rule_id": ..., "title": ..., "level": ..., "status": ...
        },
    }

`technique_sequence` / `tactic_sequence`: alerts in chronological order,
top-1 TID/tactic per alert, with consecutive duplicates collapsed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

_FINAL_CODE = Path(__file__).resolve().parent.parent.parent.parent
if str(_FINAL_CODE) not in sys.path:
    sys.path.insert(0, str(_FINAL_CODE))

from experiments.baselines.common.adapter import BaselineAdapter, BaselinePrediction
from experiments.baselines.sigma.evaluator import (
    SigmaRule, load_rules, match_event, rule_priority_key,
)

TOP_K = 5
RULES_DIR = _FINAL_CODE / "_sigma_rules" / "rules" / "windows"

# EID allow-list — only events that any Sigma category covers are worth checking,
# and many rules are noisy on common Windows utilities. We filter to those that
# the Sigma corpus actually inspects on Mordor logs.
_CHECKABLE_EIDS = {
    1, 4688,                        # process_creation
    2,                              # file_change
    3, 5156,                        # network_connection
    6,                              # driver_load
    7,                              # image_load
    8,                              # create_remote_thread
    9,                              # raw_access_thread
    10,                             # process_access
    11,                             # file_event
    12, 13, 14,                     # registry_*
    15,                             # create_stream_hash
    17, 18,                         # pipe_created
    19, 20, 21,                     # wmi_event
    22, 5158,                       # dns_query
    23, 26, 4660,                   # file_delete
    25,                             # process_tampering
    29,                             # file_rename
    400, 600, 4103, 4104,           # PowerShell
    4624, 4625, 4634, 4648, 4672,   # security: logon
    4663, 4670, 4698, 4720, 4732,   # security: object access / scheduled tasks / users
    4768, 4769, 4776,               # kerberos
    5136, 5137, 5138, 5139, 5141,   # AD changes
    7045,                           # service install
}


def _select_topk_tids(rules: list[SigmaRule], k: int) -> tuple[list[str], list[str]]:
    """Given a priority-sorted list of matching rules, dedup TIDs and tactics
    and return at most k of each (parallel order)."""
    seen: set[str] = set()
    out_tids: list[str] = []
    out_tacs: list[str] = []
    for r in rules:
        # zip TIDs with the rule's first tactic (Sigma rules tag tactics
        # separately; we associate with the rule's primary tactic).
        primary_tac = r.tactics[0] if r.tactics else ""
        for tid in r.tids:
            if tid in seen:
                continue
            seen.add(tid)
            out_tids.append(tid)
            out_tacs.append(primary_tac)
            if len(out_tids) >= k:
                return out_tids, out_tacs
    return out_tids, out_tacs


# Map from Sigma tactic slug (e.g., 'credential-access') to the human-readable
# form used in attack_flows.py / SCOPE annotations (e.g., 'Credential Access').
_TACTIC_DISPLAY = {
    "reconnaissance": "Reconnaissance",
    "resource-development": "Resource Development",
    "initial-access": "Initial Access",
    "execution": "Execution",
    "persistence": "Persistence",
    "privilege-escalation": "Privilege Escalation",
    "defense-evasion": "Defense Evasion",
    "credential-access": "Credential Access",
    "discovery": "Discovery",
    "lateral-movement": "Lateral Movement",
    "collection": "Collection",
    "command-and-control": "Command and Control",
    "exfiltration": "Exfiltration",
    "impact": "Impact",
}


def _display_tactic(slug: str) -> str:
    return _TACTIC_DISPLAY.get((slug or "").lower(), slug or "")


class SigmaAdapter(BaselineAdapter):
    """Sigma rule-based event detector."""
    name = "sigma"

    def __init__(self, rules_dir: Path | None = None, top_k: int = TOP_K):
        rules_dir = rules_dir or RULES_DIR
        print(f"  [sigma] loading rules from {rules_dir} ...")
        self.rules = load_rules(rules_dir)
        # Pre-bucket rules by category (None = unconstrained / service-based)
        # to speed up matching on hot categories. Rules without category match
        # via the per-rule logsource check.
        self._rules_by_cat: dict[str, list[SigmaRule]] = {}
        for r in self.rules:
            cat = (r.logsource.get("category") or "").lower()
            self._rules_by_cat.setdefault(cat, []).append(r)
        for v in self._rules_by_cat.values():
            v.sort(key=rule_priority_key, reverse=True)
        self.top_k = top_k
        print(f"  [sigma] {len(self.rules)} rules indexed across "
              f"{len(self._rules_by_cat)} category buckets")

    # ------------------------------------------------------------------
    def _candidate_rules(self, eid: int | None) -> Iterable[SigmaRule]:
        """Yield rules whose category COULD match this event id, plus
        all category-less (service-based) rules."""
        cats: list[str] = [""]  # always include unconstrained / service-based
        if eid in (1, 4688):
            cats.append("process_creation")
        if eid == 2:
            cats.append("file_change")
        if eid in (3, 5156):
            cats.append("network_connection")
        if eid == 6:
            cats.append("driver_load")
        if eid == 7:
            cats.append("image_load")
        if eid == 8:
            cats.append("create_remote_thread")
        if eid == 9:
            cats.append("raw_access_thread")
        if eid == 10:
            cats.append("process_access")
        if eid == 11:
            cats.append("file_event")
        if eid in (12, 13, 14):
            cats.extend(["registry_event", "registry_set",
                         "registry_add", "registry_delete", "registry_rename"])
        if eid == 15:
            cats.append("create_stream_hash")
        if eid in (17, 18):
            cats.append("pipe_created")
        if eid in (19, 20, 21):
            cats.append("wmi_event")
        if eid in (22, 5158):
            cats.append("dns_query")
        if eid in (23, 26, 4660):
            cats.append("file_delete")
        if eid == 25:
            cats.append("process_tampering")
        if eid == 29:
            cats.append("file_rename")
        if eid == 4663:
            cats.append("file_access")
        if eid == 4103:
            cats.append("ps_module")
        if eid == 4104:
            cats.append("ps_script")
        if eid in (400, 600):
            cats.append("ps_classic_start")
        for c in cats:
            for r in self._rules_by_cat.get(c, []):
                yield r

    # ------------------------------------------------------------------
    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        from pipeline.data_loader import load_and_normalize

        df = load_and_normalize(str(scenario_json_path))
        # narrow to checkable events to cut runtime by ~30-40%
        if "EventID" in df.columns:
            df = df[df["EventID"].isin(_CHECKABLE_EIDS)].reset_index(drop=False)
        df.rename(columns={"index": "_orig_index"}, inplace=True)

        per_alert: list[dict] = []
        tactic_seq: list[str] = []
        technique_seq: list[str] = []
        last_tac, last_tid = None, None
        n_evaluated = 0

        for row_pos, row in df.iterrows():
            event = {k: v for k, v in row.items()
                     if k not in ("_orig_index",) and v is not None}
            n_evaluated += 1
            cands = list(self._candidate_rules(event.get("EventID")))
            if not cands:
                continue
            hits = [r for r in cands if _eval_safe(r, event)]
            if not hits:
                continue
            hits.sort(key=rule_priority_key, reverse=True)
            topk_tids, topk_tacs = _select_topk_tids(hits, self.top_k)
            if not topk_tids:
                continue
            top_rule = hits[0]
            primary_tid = topk_tids[0]
            primary_tac = _display_tactic(topk_tacs[0])

            per_alert.append({
                "ts": str(row.get("TimeCreated") or ""),
                "event_index": int(row.get("_orig_index", row_pos)),
                "event_id": int(event.get("EventID") or 0),
                "topk_tids": topk_tids,
                "topk_tactics": [_display_tactic(t) for t in topk_tacs],
                "top_rule": {
                    "rule_id": top_rule.rule_id,
                    "title": top_rule.title,
                    "level": top_rule.level,
                    "status": top_rule.status,
                },
            })

            if primary_tid != last_tid:
                technique_seq.append(primary_tid)
                last_tid = primary_tid
            if primary_tac and primary_tac != last_tac:
                tactic_seq.append(primary_tac)
                last_tac = primary_tac

        return BaselinePrediction(
            scenario=scenario_json_path.stem,
            tactic_sequence=tactic_seq,
            technique_sequence=technique_seq,
            per_group_topk=[a["topk_tids"] for a in per_alert],  # noqa: list[list[str]] OK
            notes={
                "n_events_evaluated": int(n_evaluated),
                "n_alerts": len(per_alert),
                "alerts": per_alert,
            },
        )


def _eval_safe(rule: SigmaRule, event: dict) -> bool:
    """Wrap evaluate_rule to swallow any per-rule exception; one malformed
    rule should not abort the whole scenario."""
    from experiments.baselines.sigma.evaluator import evaluate_rule
    try:
        return evaluate_rule(rule, event)
    except Exception:
        return False
