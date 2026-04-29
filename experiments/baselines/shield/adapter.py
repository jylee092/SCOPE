"""
SHIELD baseline adapter.

End-to-end: load_and_normalize → LOF deviation analyzer → graph + tag
propagation + Louvain → per-community 3-stage CoT LLM → assemble
chronological TID sequence per scenario.

Per-community output is recorded as one "alert" in BaselinePrediction.notes
so that the existing scoring pipeline (experiments/baselines/sigma/score.py)
can be reused with a different result directory.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from experiments.baselines.common.adapter import BaselineAdapter, BaselinePrediction
from experiments.baselines.shield.deviation import filter_to_anomalous_subgraph
from experiments.baselines.shield.graph import run_graph_analyzer
from experiments.baselines.shield.llm import (
    analyze_community, CommunityAnalysis,
)


_TACTIC_DISPLAY = {
    "reconnaissance": "Reconnaissance",
    "resource development": "Resource Development",
    "initial access": "Initial Access",
    "execution": "Execution",
    "persistence": "Persistence",
    "privilege escalation": "Privilege Escalation",
    "defense evasion": "Defense Evasion",
    "credential access": "Credential Access",
    "discovery": "Discovery",
    "lateral movement": "Lateral Movement",
    "collection": "Collection",
    "command and control": "Command and Control",
    "exfiltration": "Exfiltration",
    "impact": "Impact",
}


def _norm_tactic(t: str) -> str:
    return _TACTIC_DISPLAY.get((t or "").strip().lower(), (t or "").strip())


def _community_first_ts(g, members) -> str:
    """Earliest event timestamp touching any node in the community."""
    best = ""
    for u, v, d in g.subgraph(members).edges(data=True):
        ts = (d.get("ts") or "")[:19]
        if ts and (not best or ts < best):
            best = ts
    return best


def _community_event_indices(filtered_df, members) -> list[int]:
    """Indices of rows in `filtered_df` whose ProcessGuid corresponds to any
    process node in this community. Used to align with SCOPE behavior groups
    via _orig_idx in the score step."""
    process_guids = set()
    for n in members:
        if n.startswith("P:"):
            process_guids.add(n[2:])
    if not process_guids:
        return []
    if "ProcessGuid" not in filtered_df.columns:
        return []
    mask = filtered_df["ProcessGuid"].astype(str).isin(process_guids)
    if "_orig_idx" in filtered_df.columns:
        return filtered_df.loc[mask, "_orig_idx"].tolist()
    return filtered_df.loc[mask].index.tolist()


class ShieldAdapter(BaselineAdapter):
    name = "shield"

    def __init__(self,
                 min_community_size: int = 3,
                 max_communities_per_scenario: int = 20,
                 min_confidence: float = 0.0):
        self.min_community_size = min_community_size
        self.max_communities = max_communities_per_scenario
        self.min_confidence = min_confidence
        self.api_key = config.GEMINI_API_KEY
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in config")

    # -----------------------------------------------------------------
    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        from pipeline.data_loader import load_and_normalize

        scenario_stem = scenario_json_path.stem
        t0 = time.time()
        df = load_and_normalize(str(scenario_json_path))
        # Re-attach the *normalized-DataFrame* index as a stable identifier so
        # that downstream scoring can map to SCOPE behavior groups.
        df = df.reset_index().rename(columns={"index": "_orig_idx"})

        # ---- Deviation analyzer ----
        filtered_df, lof = filter_to_anomalous_subgraph(df)

        # Size guard: SHIELD's LOF was tuned on multi-million-event DARPA
        # datasets. On small OTRF traces (<1k events), the 1-hop expansion
        # often drops the actual attack process when its parent/child are
        # not flagged as anomalous. We fall back to the full normalized df
        # whenever the filtered subgraph is too sparse to support graph
        # analysis. This adaptation is documented in §7.1.
        if filtered_df.empty or len(filtered_df) < 50 or len(df) < 1000:
            reason = ("empty" if filtered_df.empty
                      else f"too sparse (filtered={len(filtered_df)}, total={len(df)})")
            print(f"  [shield] {scenario_stem}: {reason} — using full df")
            filtered_df = df

        # ---- Graph + Louvain ----
        fallback_seeds = [
            f"P:{guid}" for guid in
            lof.anomalous_rows.get("ProcessGuid", []).dropna()
            .astype(str).unique()
        ]
        ga = run_graph_analyzer(filtered_df, fallback_seeds=fallback_seeds)
        rg = ga.reduced_graph

        comms = list(ga.communities.items())
        # Sort by size desc and limit
        comms.sort(key=lambda kv: -len(kv[1]))
        comms = comms[: self.max_communities]

        print(f"  [shield] {scenario_stem}: {lof.n_total} events → "
              f"{len(filtered_df)} filtered → {rg.number_of_nodes()} nodes "
              f"→ {len(comms)} communities")

        # ---- LLM analysis per community ----
        analyses: list[CommunityAnalysis] = []
        for cid, members in comms:
            if len(members) < self.min_community_size:
                continue
            res = analyze_community(rg, cid, members,
                                     scenario=scenario_stem,
                                     api_key=self.api_key,
                                     min_size=self.min_community_size,
                                     confidence_threshold=0.7)
            analyses.append(res)
            if not res.skipped:
                techs = [t.get("tid") for t in res.techniques][:3]
                print(f"    c{cid} ({len(members)}n): conf={res.confidence:.2f} "
                      f"stages={res.kill_chain_stages[:3]} tids={techs}")
            else:
                print(f"    c{cid} ({len(members)}n): skipped — {res.skip_reason}")

        # ---- Assemble chronological TID sequence ----
        # Order communities by earliest event timestamp.
        ts_by_comm = {a.community_id: _community_first_ts(rg, ga.communities[a.community_id])
                      for a in analyses}
        ordered = sorted(analyses,
                          key=lambda a: ts_by_comm.get(a.community_id, ""))

        per_alert: list[dict] = []
        tactic_seq: list[str] = []
        technique_seq: list[str] = []
        last_tac, last_tid = None, None

        for a in ordered:
            if a.skipped or a.confidence < self.min_confidence:
                continue
            if not a.techniques:
                continue
            ev_idxs = _community_event_indices(filtered_df,
                                                ga.communities[a.community_id])
            topk_tids: list[str] = []
            topk_tacs: list[str] = []
            seen: set[str] = set()
            for t in a.techniques:
                tid = (t.get("tid") or "").strip().upper()
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                topk_tids.append(tid)
                topk_tacs.append(_norm_tactic(t.get("tactic", "")))
                if len(topk_tids) >= 5:
                    break
            if not topk_tids:
                continue
            primary_tid = topk_tids[0]
            primary_tac = topk_tacs[0]

            per_alert.append({
                "ts": ts_by_comm.get(a.community_id, ""),
                "community_id": int(a.community_id),
                "n_members": int(a.n_members),
                "confidence": float(a.confidence),
                "topk_tids": topk_tids,
                "topk_tactics": topk_tacs,
                "kill_chain_stages": a.kill_chain_stages,
                "summary": (a.summary or "")[:500],
                "event_indices": ev_idxs,
            })

            if primary_tid != last_tid:
                technique_seq.append(primary_tid); last_tid = primary_tid
            if primary_tac and primary_tac != last_tac:
                tactic_seq.append(primary_tac); last_tac = primary_tac

        elapsed = time.time() - t0
        print(f"  [shield] {scenario_stem}: {len(per_alert)} alerts, "
              f"{elapsed:.0f}s")

        return BaselinePrediction(
            scenario=scenario_stem,
            tactic_sequence=tactic_seq,
            technique_sequence=technique_seq,
            per_group_topk=[a["topk_tids"] for a in per_alert],
            notes={
                "n_total_events": int(lof.n_total),
                "n_filtered_events": int(len(filtered_df)),
                "n_lof_anomalous": int(lof.n_anomalous),
                "n_communities": len(comms),
                "n_alerts": len(per_alert),
                "alerts": per_alert,
                "elapsed_sec": round(elapsed, 1),
            },
        )
