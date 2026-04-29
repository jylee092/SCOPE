"""
Section 5. Probabilistic Attack Chain Inference

 - Multi-dimensional transition (tactical + semantic + causal)
 - Top-K Viterbi with hole-bridging (Algorithm 1)
 - Campaign-level novelty scoring

--------
sort_results_by_time(results, final_df)
load_tactic_map(mitre_csv_path)
build_group_nodes(sorted_results, tactic_map, features_by_gid=None)
load_campaign_library(campaign_folder, tactic_map)
topk_viterbi(group_nodes, scorer, beam_k, max_skip, skip_penalty,
             transition_weight, campaigns) -> ViterbiResult
print_viterbi_report(result)
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def sort_results_by_time(results: list[dict], final_df: pd.DataFrame) -> list[dict]:
    """analyze() ...anchor TimeCreated ..."""
    enriched = []
    for r in results:
        try:
            anchor_idx = int(r["group_id"].rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            anchor_idx = None

        anchor_time = (
            final_df.loc[anchor_idx, "TimeCreated"]
            if anchor_idx is not None and anchor_idx in final_df.index
            else None
        )
        enriched.append({**r, "anchor_idx": anchor_idx, "anchor_time": anchor_time})

    enriched.sort(key=lambda x: (x["anchor_time"] is None, x["anchor_time"]))
    return enriched


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def load_tactic_map(mitre_csv_path: str) -> dict:
    """MITRE CSV...technique_id → [tactic, ...] ..."""
    df = pd.read_csv(mitre_csv_path)

    id_col = next(
        (c for c in df.columns if c.lower() in ("technique_id", "id", "techniqueid")),
        None,
    )
    tactic_col = next((c for c in df.columns if "tactic" in c.lower()), None)
    if not id_col or not tactic_col:
        raise ValueError(f"...: {list(df.columns)}")

    tactic_map: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        tid = str(row[id_col]).strip()
        raw = str(row[tactic_col]) if pd.notna(row[tactic_col]) else ""
        if not tid or not raw or raw == "nan":
            continue
        tactics = [t.strip() for t in re.split(r"[,;]", raw) if t.strip()]
        tactic_map[tid] = tactics

    print(f"  Tactic ...: {len(tactic_map)}...")
    return tactic_map


# ──────────────────────────────────────────────────────────────────────────────
# (B) Tactical Flow Compatibility -- P_tac (Table 2, Section 4.4)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TransitionResult:
    from_tactic: str
    to_tactic:   str
    weight:      float
    rule:        str
    rule_name:   str
    note:        str = ""


class TacticalScorer:
    """R1~R9 ...Tactic ...-- M_tac ..."""

    _TACTICS = [
        "Reconnaissance", "Resource Development", "Initial Access",
        "Execution", "Persistence", "Privilege Escalation",
        "Defense Evasion", "Credential Access", "Discovery",
        "Lateral Movement", "Collection", "Command and Control",
        "Exfiltration", "Impact",
    ]
    _RULES = {
        "R9":  {"name": "Forbidden",           "weight": 0.02},
        "R2":  {"name": "Primary Flow",        "weight": 1.0 },
        "R3A": {"name": "Post-Exploit Loop A", "weight": 0.9 },
        "R3B": {"name": "Post-Exploit Loop B", "weight": 0.8 },
        "R4":  {"name": "Wildcard IN",         "weight": 0.8 },
        "R5":  {"name": "Wildcard OUT",        "weight": 0.7 },
        "R5B": {"name": "Wildcard OUT Rev",    "weight": 0.4 },
        "R1":  {"name": "Self-Loop",           "weight": 0.5 },
        "R6":  {"name": "Forward Skip",        "weight": 0.5 },
        "R7":  {"name": "Far Skip",            "weight": 0.3 },
        "R8A": {"name": "Backward Short",      "weight": 0.1 },
        "R8B": {"name": "Backward Long",       "weight": 0.05},
    }
    # fallback = _RULES['R1'] weight 0.5
    _SELF_LOOP_WEIGHTS = {
        "Execution":       0.40,
        "Defense Evasion": 0.60,
    }
    _WILDCARD_IN_WEIGHTS = {
        "Defense Evasion": 0.70,
    }
    _PRIMARY = {
        ("Reconnaissance",       "Resource Development"),
        ("Resource Development", "Initial Access"),
        ("Initial Access",       "Execution"),
        ("Execution",            "Persistence"),
        ("Persistence",          "Privilege Escalation"),
        ("Privilege Escalation", "Defense Evasion"),
        ("Lateral Movement",     "Collection"),
        ("Collection",           "Command and Control"),
        ("Command and Control",  "Exfiltration"),
        ("Exfiltration",         "Impact"),
        ("Execution",            "Privilege Escalation"),
        ("Execution",            "Credential Access"),
    }
    _PE_LOOP_DIRECT = {
        ("Credential Access", "Discovery"),        ("Discovery",        "Credential Access"),
        ("Credential Access", "Lateral Movement"), ("Lateral Movement", "Credential Access"),
        ("Discovery",         "Lateral Movement"), ("Lateral Movement", "Discovery"),
    }
    _PE_LOOP_COLLECT = {
        ("Credential Access", "Collection"),
        ("Discovery",         "Collection"),
        ("Lateral Movement",  "Collection"),
    }
    _WILDCARD      = {"Defense Evasion", "Privilege Escalation"}
    _FORBIDDEN_SRC = {"Impact"}
    _FORBIDDEN_PAIRS = {
        ("Exfiltration", t) for t in [
            "Reconnaissance", "Resource Development", "Initial Access",
            "Execution", "Persistence", "Privilege Escalation", "Defense Evasion",
        ]
    }
    _OVERRIDES = {
        ("Execution",   "Privilege Escalation"): ("R2", 1.0, "R2/R4 ...R2 ..."),
        ("Execution",   "Defense Evasion"):      ("R2", 1.0, "R2/R4 ...R2 ..."),
        ("Persistence", "Privilege Escalation"): ("R2", 1.0, "R2/R4 ...R2 ..."),
    }

    def __init__(self, config_path=None, anomaly_threshold: float = 0.1):
        self.anomaly_threshold = anomaly_threshold
        cfg_path = Path(config_path) if config_path else None

        if cfg_path and cfg_path.exists():
            self._load_from_file(cfg_path)
        else:
            self._tactic_order    = {t: i for i, t in enumerate(self._TACTICS)}
            self._rules           = self._RULES
            self._primary         = self._PRIMARY
            self._pe_direct       = self._PE_LOOP_DIRECT
            self._pe_collect      = self._PE_LOOP_COLLECT
            self._wildcard        = self._WILDCARD
            self._forbidden_src   = self._FORBIDDEN_SRC
            self._forbidden_pairs = self._FORBIDDEN_PAIRS
            self._overrides       = self._OVERRIDES
            self._self_loop_w     = self._SELF_LOOP_WEIGHTS
            self._wildcard_in_w   = self._WILDCARD_IN_WEIGHTS

    def _load_from_file(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        self._tactic_order    = {t: i for i, t in enumerate(cfg["tactics"])}
        self._rules           = {r["id"]: r for r in cfg["rules"]}
        self._primary         = {(a, b) for a, b in cfg["primary_flow"]}
        self._pe_direct       = {(a, b) for a, b in cfg["post_exploit_loop_direct"]}
        self._pe_collect      = {(a, b) for a, b in cfg["post_exploit_loop_collect"]}
        self._wildcard        = set(cfg["wildcard_tactics"])
        self._forbidden_src   = set(cfg["forbidden_sources"])
        self._forbidden_pairs = {(a, b) for a, b in cfg["forbidden_pairs"]}
        self._overrides       = {
            (o["from"], o["to"]): (o["rule"], o["weight"], o.get("note", ""))
            for o in cfg.get("overrides", [])
        }
        self._self_loop_w     = dict(cfg.get("self_loop_weights",   self._SELF_LOOP_WEIGHTS))
        self._wildcard_in_w   = dict(cfg.get("wildcard_in_weights", self._WILDCARD_IN_WEIGHTS))

    def score(self, from_tactic: str, to_tactic: str) -> TransitionResult:
        def make(rule_id, note=""):
            r = self._rules[rule_id]
            return TransitionResult(from_tactic, to_tactic, r["weight"], rule_id, r["name"], note)

        pair = (from_tactic, to_tactic)

        if pair in self._overrides:
            rule_id, w, note = self._overrides[pair]
            r = self._rules[rule_id]
            return TransitionResult(from_tactic, to_tactic, w, rule_id, r["name"], note)

        if from_tactic == to_tactic:
            # per-tactic self-loop override (v17); fallback = R1 weight.
            w = self._self_loop_w.get(from_tactic)
            if w is not None:
                return TransitionResult(from_tactic, to_tactic, w, "R1",
                                        self._rules["R1"]["name"],
                                        f"self-loop({from_tactic})")
            return make("R1", "...Tactic ...")

        fi = self._tactic_order.get(from_tactic, -1)
        ti = self._tactic_order.get(to_tactic,   -1)
        if fi == -1 or ti == -1:
            return TransitionResult(from_tactic, to_tactic, 0.3, "R7", "Far Skip", "Unknown Tactic")

        diff = ti - fi

        if from_tactic in self._forbidden_src:
            return make("R9", f"{from_tactic} ...")
        if pair in self._forbidden_pairs:
            return make("R9", "Exfiltration ...")
        if pair in self._primary:
            return make("R2", "ATT&CK Kill Chain ...")
        if pair in self._pe_direct:
            return make("R3A", "CA / Discovery / Lateral Movement ...")
        if pair in self._pe_collect:
            return make("R3B", "Post-Exploit Loop → Collection ...")

        is_from_wc = from_tactic in self._wildcard
        is_to_wc   = to_tactic   in self._wildcard
        if not is_from_wc and is_to_wc:
            w = self._wildcard_in_w.get(to_tactic)
            if w is not None:
                return TransitionResult(from_tactic, to_tactic, w, "R4",
                                        self._rules["R4"]["name"],
                                        f"wildcard-in→{to_tactic}")
            return make("R4", f"...{to_tactic}...")
        if is_from_wc and not is_to_wc:
            return make("R5",  f"{from_tactic} → ...") if diff > 0 \
              else make("R5B", f"{from_tactic} → ...")

        if diff > 0:
            return make("R6", f"...{diff}...") if diff <= 2 \
              else make("R7", f"...{diff}...")
        abs_diff = abs(diff)
        return make("R8A", f"...{abs_diff}...") if abs_diff <= 2 \
          else make("R8B", f"...{abs_diff}...")


# backward compat alias
TransitionScorer = TacticalScorer


# ──────────────────────────────────────────────────────────────────────────────
# (C) Semantic Continuity -- P_sem (Section 4.4, Eq. 4)
# ──────────────────────────────────────────────────────────────────────────────
_SEM_SCORER_CACHE: dict[str, "SemanticScorer"] = {}


class SemanticScorer:
    """Semantic ...Cross-encoder ...bi-encoder ...

    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        backend: str = "cross-encoder",
        calibration: str = "linear",
        sigmoid_center: float = 0.5,
        sigmoid_scale: float = 8.0,
    ):
        self._backend = backend
        self._calibration = calibration
        self._sig_c0 = float(sigmoid_center)
        self._sig_beta = float(sigmoid_scale)
        if backend == "cross-encoder":
            from sentence_transformers import CrossEncoder
            print(f"  [cache-miss] CrossEncoder ...: {model_name}")
            self._model = CrossEncoder(model_name)
        elif backend == "bi-encoder":
            from sentence_transformers import SentenceTransformer
            print(f"  [cache-miss] SentenceTransformer (bi-encoder) ...: {model_name}")
            self._model = SentenceTransformer(model_name)
            self._emb_cache: dict[str, object] = {}
        else:
            raise ValueError(f"...semantic backend: {backend}")
        self._cache: dict[tuple, float] = {}

    def _encode(self, text: str):
        cached = self._emb_cache.get(text)
        if cached is not None:
            return cached
        import numpy as np
        vec = self._model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        vec = np.asarray(vec, dtype="float32")
        self._emb_cache[text] = vec
        return vec

    def score(self, desc_i: str, desc_j: str) -> float:
        key = (desc_i[:200], desc_j[:200])
        if key in self._cache:
            return self._cache[key]

        if self._backend == "cross-encoder":
            raw = float(self._model.predict([(desc_i, desc_j)])[0])
            s = 1.0 / (1.0 + math.exp(-raw))
        else:  # bi-encoder
            ei = self._encode(desc_i)
            ej = self._encode(desc_j)
            cos = float((ei * ej).sum())

            if self._calibration == "sigmoid":
                s = 1.0 / (1.0 + math.exp(-self._sig_beta * (cos - self._sig_c0)))
            else:
                # legacy linear: cos ∈ [-1,1] → [0,1].
                s = (cos + 1.0) * 0.5
            s = max(_EPS, min(1.0, s))

        self._cache[key] = s
        return s


def get_semantic_scorer(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    backend: str = "cross-encoder",
    calibration: str = "linear",
    sigmoid_center: float = 0.5,
    sigmoid_scale: float = 8.0,
) -> "SemanticScorer":
    """...SemanticScorer ...model+backend+calibration ..."""
    key = (model_name, backend, calibration, sigmoid_center, sigmoid_scale)
    if key not in _SEM_SCORER_CACHE:
        _SEM_SCORER_CACHE[key] = SemanticScorer(
            model_name, backend=backend,
            calibration=calibration,
            sigmoid_center=sigmoid_center,
            sigmoid_scale=sigmoid_scale,
        )
    return _SEM_SCORER_CACHE[key]


# ──────────────────────────────────────────────────────────────────────────────
# (D) Causal Lineage -- P_cau (Section 4.4, Eq. 5)
# ──────────────────────────────────────────────────────────────────────────────
_TACTIC_IO: dict[str, tuple[set, set]] = {
    "Reconnaissance":       (set(),                       {"network"}),
    "Resource Development": (set(),                       {"file", "network"}),
    "Initial Access":       ({"network"},                 {"process"}),
    "Execution":            ({"process", "file"},         {"process"}),
    "Persistence":          ({"process"},                 {"file", "registry", "service"}),
    "Privilege Escalation": ({"process"},                 {"process", "user"}),
    "Defense Evasion":      ({"process", "file"},         {"process", "file", "registry"}),
    "Credential Access":    ({"process"},                 {"user", "file"}),
    "Discovery":            ({"process"},                 {"process"}),
    "Lateral Movement":     ({"user", "network"},         {"process", "network"}),
    "Collection":           ({"process", "file"},         {"file"}),
    "Command and Control":  ({"process"},                 {"network"}),
    "Exfiltration":         ({"file", "network"},         {"network"}),
    "Impact":               ({"process"},                 {"file", "process", "service"}),
}

_EPS = 1e-6


def extract_entity_types(features: dict) -> set[str]:
    """...feature dict...entity type ..."""
    types: set[str] = set()
    f = features.get("features", features)

    ctx = f.get("execution_context") or {}
    if ctx.get("process_chains"):
        types.add("process")

    cmd = f.get("command_script") or {}
    if cmd.get("entries"):
        types.add("process")

    per = f.get("persistence") or {}
    if per.get("registry_signals"):
        types.add("registry")
    if per.get("dropped_files"):
        types.add("file")

    net = f.get("network") or {}
    if net.get("connections"):
        types.add("network")

    idn = f.get("identity") or {}
    if idn.get("integrity_level") or idn.get("user"):
        types.add("user")

    return types


class CausalScorer:
    """Entity type overlap ...causal lineage ...Eq. 5).

    """

    def __init__(self, technique_io: Optional[dict] = None):
        self._tech_io = technique_io or {}

    def _get_io(self, technique_id: str, tactic: str) -> tuple[set[str], set[str]]:
        """technique → In/Out. ...tactic fallback."""
        t = self._tech_io.get(technique_id)
        if t:
            return set(t["in"]), set(t["out"])
        fallback = _TACTIC_IO.get(tactic, (set(), set()))
        return fallback

    def score(
        self,
        tech_i: str, tactic_i: str,
        tech_j: str, tactic_j: str,
        entities_i: set[str], entities_j: set[str],
    ) -> float:
        _, out_i = self._get_io(tech_i, tactic_i)
        in_j, _  = self._get_io(tech_j, tactic_j)
        shared_observed = entities_i & entities_j
        overlap = out_i & in_j & shared_observed
        return len(overlap) / (len(in_j) + _EPS)


# ──────────────────────────────────────────────────────────────────────────────
# (E) Adaptive Multi-Dimensional Fusion -- P_trans (Section 4.4, Eq. 6)
# ──────────────────────────────────────────────────────────────────────────────
class MultiDimTransitionScorer:
    """P_trans = exp( Σ w_k · log P_k )  with data-adaptive weights."""

    def __init__(
        self,
        tac_scorer: TacticalScorer,
        sem_scorer: Optional[SemanticScorer] = None,
        cau_scorer: Optional[CausalScorer] = None,
        w_tac: float = 0.5,
        w_sem: float = 0.3,
        w_cau: float = 0.2,
        self_loop_tid_penalty: float = 1.0,
    ):
        self.tac = tac_scorer
        self.sem = sem_scorer
        self.cau = cau_scorer
        self._w_tac = w_tac
        self._w_sem = w_sem
        self._w_cau = w_cau
        self._self_loop_tid_penalty = self_loop_tid_penalty

    def score(
        self,
        cand_i: "Candidate", cand_j: "Candidate",
        node_i: "GroupNode", node_j: "GroupNode",
    ) -> tuple[float, TransitionResult]:
        """Returns (fused_score, tactical_result)."""
        tac_result = self.tac.score(cand_i.tactic, cand_j.tactic)
        p_tac = max(tac_result.weight, _EPS)

        w_tac = self._w_tac
        w_sem = self._w_sem
        w_cau = self._w_cau

        # --- Semantic ---
        if self.sem and node_i.description and node_j.description:
            text_i = node_i.description + " " + cand_i.mitre_description
            text_j = node_j.description + " " + cand_j.mitre_description
            p_sem = max(self.sem.score(text_i, text_j), _EPS)
        else:
            p_sem = 1.0
            w_sem = 0.0

        # --- Causal (adaptive weight via entity coverage ρ) ---
        if self.cau and node_i.entity_types and node_j.entity_types:
            p_cau = max(
                self.cau.score(
                    cand_i.technique_id, cand_i.tactic,
                    cand_j.technique_id, cand_j.tactic,
                    node_i.entity_types, node_j.entity_types,
                ),
                _EPS,
            )
            in_j, _ = self.cau._get_io(cand_j.technique_id, cand_j.tactic)
            rho = (
                len(node_i.entity_types & node_j.entity_types) / max(len(in_j), 1)
            )
            if rho < 0.3:
                w_cau *= rho / 0.3
        else:
            p_cau = 1.0
            w_cau = 0.0

        # --- Renormalize ---
        total_w = w_tac + w_sem + w_cau
        if total_w < _EPS:
            return p_tac, tac_result
        w_tac /= total_w
        w_sem /= total_w
        w_cau /= total_w

        # --- Geometric mean (Eq. 6) ---
        log_fused = (
            w_tac * math.log(p_tac)
            + w_sem * math.log(p_sem)
            + w_cau * math.log(p_cau)
        )
        fused = math.exp(log_fused)

        if (cand_i.technique_id == cand_j.technique_id
                and self._self_loop_tid_penalty < 1.0):
            fused *= self._self_loop_tid_penalty

        return fused, tac_result


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Candidate:
    rank:              int
    technique_id:      str
    technique_name:    str
    tactic:            str
    similarity:        float
    log_emission:      float
    mitre_description: str = ""


@dataclass
class GroupNode:
    group_id:     str
    technique_id: str
    anchor_time:  object
    confidence:   float
    candidates:   list          # list[Candidate]
    description:  str = ""      # LLM-generated description
    entity_types: set = field(default_factory=set)
    conf_margin:  float = 0.0

    sim_margin:   float = 0.0



@dataclass
class ViterbiResult:
    groups:           list      # list[GroupNode]
    best_path:        list      # list[Candidate]
    best_score:       float
    score_breakdown:  list      # list[dict]
    novelty_score:    float = 0.0
    closest_campaign: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def build_group_nodes(
    sorted_results: list[dict],
    tactic_map: dict,
    features_by_gid: Optional[dict] = None,
) -> list[GroupNode]:
    """
    """
    def resolve_tactic(tid: str) -> str:
        tactics = tactic_map.get(tid)
        if tactics:
            return tactics[0]
        parent = tid.split(".")[0] if "." in tid else None
        if parent:
            tactics = tactic_map.get(parent)
            if tactics:
                return tactics[0]
        return "Unknown"

    nodes: list[GroupNode] = []
    for r in sorted_results:
        candidates: list[Candidate] = []
        for s in r.get("similar_techniques", []):
            sim = max(s["similarity"], 1e-9)
            p_ttp = s.get("p_ttp", sim)
            p_ttp = max(p_ttp, 1e-9)
            candidates.append(Candidate(
                rank              = s["rank"],
                technique_id      = s["technique_id"],
                technique_name    = s["technique_name"],
                tactic            = resolve_tactic(s["technique_id"]),
                similarity        = sim,
                log_emission      = math.log(p_ttp),
                mitre_description = s.get("description", ""),
            ))

        gid = r["group_id"]
        feat = (features_by_gid or {}).get(gid, {})
        entity_types = extract_entity_types(feat) if feat else set()

        sim_margin = 0.0
        if len(candidates) >= 2:
            sim_margin = float(candidates[0].similarity - candidates[1].similarity)
        elif len(candidates) == 1:
            sim_margin = float(candidates[0].similarity)

        nodes.append(GroupNode(
            group_id     = gid,
            technique_id = r["technique_id"],
            anchor_time  = r.get("anchor_time"),
            confidence   = r.get("confidence", 1.0),
            candidates   = candidates,
            description  = r.get("generated_description", ""),
            entity_types = entity_types,
            conf_margin  = float(r.get("confidence_margin", 0.0)),
            sim_margin   = sim_margin,
        ))
    return nodes


# ──────────────────────────────────────────────────────────────────────────────
# (H) Campaign Library + Novelty (Section 4.5)
# ──────────────────────────────────────────────────────────────────────────────
def load_campaign_library(
    campaign_folder: str | Path,
    tactic_map: dict,
) -> list[dict]:
    """Campaign JSON ...tactic ..."""
    folder = Path(campaign_folder)
    if not folder.exists():
        print(f"  Campaign ...: {folder}")
        return []

    campaigns: list[dict] = []
    for path in sorted(folder.glob("*-enterprise-layer.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tech_ids = [
            t["techniqueID"]
            for t in data.get("techniques", [])
            if t.get("score", 0) > 0
        ]
        tactics: list[str] = []
        seen: set[str] = set()
        for tid in tech_ids:
            t_list = tactic_map.get(tid, [])
            for tac in t_list:
                if tac not in seen:
                    seen.add(tac)
                    tactics.append(tac)

        if tactics:
            campaigns.append({
                "name": data.get("name", path.stem),
                "techniques": tech_ids,
                "tactics": tactics,
            })

    print(f"  Campaign ...: {len(campaigns)}...")
    return campaigns


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest Common Subsequence ..."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_novelty(
    pred_tactics: list[str],
    campaigns: list[dict],
) -> tuple[float, str]:
    """n(S*) = 1 - max_C LCS(S*, C) / |S*|. Returns (novelty, closest_name)."""
    if not pred_tactics or not campaigns:
        return 1.0, ""
    max_lcs = 0
    closest = ""
    for c in campaigns:
        lcs = _lcs_length(pred_tactics, c["tactics"])
        if lcs > max_lcs:
            max_lcs = lcs
            closest = c["name"]
    novelty = round(1 - max_lcs / len(pred_tactics), 4)
    return novelty, closest


# ──────────────────────────────────────────────────────────────────────────────
# (I) Top-K Viterbi with Hole-Bridging (Algorithm 1, Eq. 5)
# ──────────────────────────────────────────────────────────────────────────────
_LOG_ZERO = -1e9


def topk_viterbi(
    group_nodes: list[GroupNode],
    scorer: MultiDimTransitionScorer,
    beam_k: int = 5,
    max_skip: int = 2,
    skip_penalty: float = 0.5,
    transition_weight: float = 0.5,
    campaigns: Optional[list[dict]] = None,
    margin_gated: bool = False,
    margin_low: float = 0.05,
    margin_high: float = 0.20,
    alpha_low_margin: float = 0.40,
    alpha_high_margin: float = 0.05,
    sim_gated: bool = False,
    sim_margin_low: float = 0.03,
    sim_margin_high: float = 0.10,
    alpha_low_sim: float = 0.5,
    alpha_high_sim: float = 0.1,
    hard_tactic_filter: bool = False,
) -> ViterbiResult:
    """
    Top-K Viterbi with hole-bridging.

    score = Σ [ (1-α)·log P_ttp + α·log P_trans + d·log λ ]
    α = transition_weight, λ = skip_penalty, D = max_skip

    Margin-gated α (optional, margin_gated=True):
        per-node α depending on GroupNode.conf_margin:
          margin < margin_low   → α = alpha_low_margin   (ambiguous → let transition decide)
          margin > margin_high  → α = alpha_high_margin  (clear → trust FAISS)
          otherwise             → α = transition_weight  (default fallback)

    Algorithm 1 from the paper:
    - δ[s][t] stores the best cumulative score at step s, candidate t
    - ψ[s][t] stores the backpointer (prev_step, prev_technique_id)
    - d > 0 bridges unobserved intermediate steps with penalty λ^d
    """
    α  = transition_weight
    T  = len(group_nodes)
    if T == 0:
        raise ValueError("group_nodes...")

    def alpha_for(node: GroupNode) -> float:
        if sim_gated:
            sm = node.sim_margin
            if sm < sim_margin_low:
                return alpha_low_sim
            if sm > sim_margin_high:
                return alpha_high_sim
            return α
        if margin_gated:
            m = node.conf_margin
            if m < margin_low:
                return alpha_low_margin
            if m > margin_high:
                return alpha_high_margin
            return α
        return α

    log_λ = math.log(max(skip_penalty, _EPS))

    # δ[step][technique_id] = score
    # ψ[step][technique_id] = (prev_step, prev_technique_id) or None
    # cand_lookup[step][technique_id] = Candidate
    delta: list[dict[str, float]] = [{} for _ in range(T)]
    psi:   list[dict[str, Optional[tuple]]] = [{} for _ in range(T)]
    cand_lookup: list[dict[str, Candidate]] = [{} for _ in range(T)]

    # --- Initialization (s=0) ---
    a0 = alpha_for(group_nodes[0])
    for c in group_nodes[0].candidates[:beam_k]:
        delta[0][c.technique_id] = (1 - a0) * c.log_emission
        psi[0][c.technique_id] = None
        cand_lookup[0][c.technique_id] = c

    # --- Recurrence (s=1..T-1) ---
    for s in range(1, T):
        node_j = group_nodes[s]
        a_j = alpha_for(node_j)
        ew_j = 1 - a_j
        for cand_j in node_j.candidates[:beam_k]:
            best_score = _LOG_ZERO
            best_back: Optional[tuple] = None

            for d in range(0, min(max_skip, s) + 1):
                prev_s = s - 1 - d
                if prev_s < 0:
                    break
                node_i = group_nodes[prev_s]

                for cand_i_tid, prev_score in delta[prev_s].items():
                    cand_i = cand_lookup[prev_s][cand_i_tid]
                    fused, tac_res = scorer.score(cand_i, cand_j, node_i, node_j)
                    if hard_tactic_filter and tac_res.rule == "R9":
                        continue
                    log_trans = math.log(max(fused, _EPS))

                    v = (prev_score
                         + d * log_λ
                         + ew_j * cand_j.log_emission
                         + a_j * log_trans)

                    if v > best_score:
                        best_score = v
                        best_back = (prev_s, cand_i_tid)

            delta[s][cand_j.technique_id] = best_score
            psi[s][cand_j.technique_id] = best_back
            cand_lookup[s][cand_j.technique_id] = cand_j

    # --- Backtrace ---
    best_tid = max(delta[T - 1], key=lambda t: delta[T - 1][t])
    best_score = delta[T - 1][best_tid]

    path_indices: list[tuple[int, str]] = []
    cur_s, cur_tid = T - 1, best_tid
    while cur_s is not None:
        path_indices.append((cur_s, cur_tid))
        back = psi[cur_s].get(cur_tid)
        if back is None:
            break
        cur_s, cur_tid = back
    path_indices.reverse()

    best_path = [cand_lookup[s][tid] for s, tid in path_indices]

    # --- Score Breakdown ---
    breakdown: list[dict] = []
    for idx, (s, tid) in enumerate(path_indices):
        c = cand_lookup[s][tid]
        entry = {
            "step":           idx + 1,
            "group_idx":      s,
            "group_id":       group_nodes[s].group_id,
            "technique_id":   c.technique_id,
            "technique_name": c.technique_name,
            "tactic":         c.tactic,
            "similarity":     c.similarity,
            "log_emission":   round(c.log_emission, 4),
        }
        if idx > 0:
            prev_s, prev_tid = path_indices[idx - 1]
            prev_c = cand_lookup[prev_s][prev_tid]
            fused, tac_result = scorer.score(
                prev_c, c, group_nodes[prev_s], group_nodes[s]
            )
            skip_d = s - prev_s - 1
            entry["transition_from"]   = prev_c.tactic
            entry["transition_weight"] = round(fused, 4)
            entry["transition_rule"]   = tac_result.rule
            entry["transition_note"]   = tac_result.note
            entry["log_transition"]    = round(math.log(max(fused, _EPS)), 4)
            entry["skip_distance"]     = skip_d
        else:
            entry["transition_from"]   = None
            entry["transition_weight"] = None
            entry["transition_rule"]   = None
            entry["transition_note"]   = None
            entry["log_transition"]    = None
            entry["skip_distance"]     = 0
        breakdown.append(entry)

    # --- Novelty ---
    pred_tactics = [c.tactic for c in best_path]
    novelty = 0.0
    closest = ""
    if campaigns:
        novelty, closest = compute_novelty(pred_tactics, campaigns)

    return ViterbiResult(
        groups           = group_nodes,
        best_path        = best_path,
        best_score       = round(best_score, 4),
        score_breakdown  = breakdown,
        novelty_score    = novelty,
        closest_campaign = closest,
    )


def topk_posterior_decode(
    group_nodes: list[GroupNode],
    scorer: MultiDimTransitionScorer,
    beam_k: int = 5,
    max_skip: int = 0,
    skip_penalty: float = 0.25,
    transition_weight: float = 0.5,
    campaigns: Optional[list[dict]] = None,
    hard_tactic_filter: bool = False,
) -> ViterbiResult:
    """Forward-Backward max-marginal decoding.



    """
    α = transition_weight
    T = len(group_nodes)
    if T == 0:
        raise ValueError("group_nodes...")
    log_λ = math.log(max(skip_penalty, _EPS))

    # shared candidate lookup
    cand_lookup: list[dict[str, Candidate]] = [{} for _ in range(T)]
    for s in range(T):
        for c in group_nodes[s].candidates[:beam_k]:
            cand_lookup[s][c.technique_id] = c

    # ── Forward pass (same structure as Viterbi δ) ──────────────────────────
    forward: list[dict[str, float]] = [{} for _ in range(T)]
    for c in group_nodes[0].candidates[:beam_k]:
        forward[0][c.technique_id] = (1 - α) * c.log_emission
    for s in range(1, T):
        node_j = group_nodes[s]
        for cand_j in node_j.candidates[:beam_k]:
            best = _LOG_ZERO
            for d in range(0, min(max_skip, s) + 1):
                prev_s = s - 1 - d
                if prev_s < 0:
                    break
                node_i = group_nodes[prev_s]
                for cand_i_tid, prev_score in forward[prev_s].items():
                    cand_i = cand_lookup[prev_s][cand_i_tid]
                    fused, tac_res = scorer.score(cand_i, cand_j, node_i, node_j)
                    if hard_tactic_filter and tac_res.rule == "R9":
                        continue
                    log_trans = math.log(max(fused, _EPS))
                    v = (prev_score
                         + d * log_λ
                         + (1 - α) * cand_j.log_emission
                         + α * log_trans)
                    if v > best:
                        best = v
            forward[s][cand_j.technique_id] = best

    backward: list[dict[str, float]] = [{} for _ in range(T)]
    for tid in forward[T - 1]:
        backward[T - 1][tid] = 0.0
    for s in range(T - 2, -1, -1):
        node_i = group_nodes[s]
        for cand_i_tid in forward[s].keys():
            cand_i = cand_lookup[s][cand_i_tid]
            best = _LOG_ZERO
            for d in range(0, min(max_skip, T - 1 - s) + 1):
                next_s = s + 1 + d
                if next_s >= T:
                    break
                node_j = group_nodes[next_s]
                for cand_j_tid, next_score in backward[next_s].items():
                    cand_j = cand_lookup[next_s][cand_j_tid]
                    fused, tac_res = scorer.score(cand_i, cand_j, node_i, node_j)
                    if hard_tactic_filter and tac_res.rule == "R9":
                        continue
                    log_trans = math.log(max(fused, _EPS))
                    v = (next_score
                         + d * log_λ
                         + (1 - α) * cand_j.log_emission
                         + α * log_trans)
                    if v > best:
                        best = v
            backward[s][cand_i_tid] = best

    # ── Per-step argmax of marginal ─────────────────────────────────────────
    best_path: list[Candidate] = []
    total_score = 0.0
    for s in range(T):
        best_tid = None
        best_m = _LOG_ZERO
        for tid, f_val in forward[s].items():
            m = f_val + backward[s].get(tid, _LOG_ZERO)
            if m > best_m:
                best_m = m
                best_tid = tid
        if best_tid is None:
            best_tid = group_nodes[s].candidates[0].technique_id
            cand_lookup[s].setdefault(best_tid, group_nodes[s].candidates[0])
        best_path.append(cand_lookup[s][best_tid])
        total_score = max(total_score, best_m)

    # ── Score breakdown ─────────────────────────────────────────────────────
    breakdown: list[dict] = []
    for s, c in enumerate(best_path):
        entry = {
            "step":           s + 1,
            "group_idx":      s,
            "group_id":       group_nodes[s].group_id,
            "technique_id":   c.technique_id,
            "technique_name": c.technique_name,
            "tactic":         c.tactic,
            "similarity":     c.similarity,
            "log_emission":   round(c.log_emission, 4),
        }
        if s > 0:
            prev = best_path[s - 1]
            fused, tac_res = scorer.score(prev, c, group_nodes[s - 1], group_nodes[s])
            entry["transition_from"]   = prev.tactic
            entry["transition_weight"] = round(fused, 4)
            entry["transition_rule"]   = tac_res.rule
            entry["transition_note"]   = tac_res.note
            entry["log_transition"]    = round(math.log(max(fused, _EPS)), 4)
            entry["skip_distance"]     = 0
        else:
            entry.update({
                "transition_from": None, "transition_weight": None,
                "transition_rule": None, "transition_note": None,
                "log_transition": None, "skip_distance": 0,
            })
        breakdown.append(entry)

    pred_tactics = [c.tactic for c in best_path]
    novelty = 0.0
    closest = ""
    if campaigns:
        novelty, closest = compute_novelty(pred_tactics, campaigns)

    return ViterbiResult(
        groups           = group_nodes,
        best_path        = best_path,
        best_score       = round(total_score, 4),
        score_breakdown  = breakdown,
        novelty_score    = novelty,
        closest_campaign = closest,
    )


def _logsumexp(values):
    """...logsumexp over iterable of log-space values."""
    values = [v for v in values if v > _LOG_ZERO]
    if not values:
        return _LOG_ZERO
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def topk_sumproduct_decode(
    group_nodes: list[GroupNode],
    scorer: MultiDimTransitionScorer,
    beam_k: int = 5,
    max_skip: int = 0,
    skip_penalty: float = 0.25,
    transition_weight: float = 0.5,
    campaigns: Optional[list[dict]] = None,
    hard_tactic_filter: bool = False,
) -> ViterbiResult:
    """Sum-product (classical forward-backward) posterior decoding.


    log-space:
      forward[s][t]  = log Σ_{paths 0..s ending at t} exp(score)
      backward[s][t] = log Σ_{paths s..T-1 starting at t} exp(score_after_s)
      posterior[s][t] ∝ forward[s][t] + backward[s][t]
      per-step pick = argmax_t posterior[s][t]

    """
    α = transition_weight
    T = len(group_nodes)
    if T == 0:
        raise ValueError("group_nodes...")
    log_λ = math.log(max(skip_penalty, _EPS))

    cand_lookup: list[dict[str, Candidate]] = [{} for _ in range(T)]
    for s in range(T):
        for c in group_nodes[s].candidates[:beam_k]:
            cand_lookup[s][c.technique_id] = c

    # ── Forward pass (sum-product) ──────────────────────────────────────────
    forward: list[dict[str, float]] = [{} for _ in range(T)]
    for c in group_nodes[0].candidates[:beam_k]:
        forward[0][c.technique_id] = (1 - α) * c.log_emission
    for s in range(1, T):
        node_j = group_nodes[s]
        for cand_j in node_j.candidates[:beam_k]:
            contribs = []
            for d in range(0, min(max_skip, s) + 1):
                prev_s = s - 1 - d
                if prev_s < 0:
                    break
                node_i = group_nodes[prev_s]
                for cand_i_tid, prev_score in forward[prev_s].items():
                    cand_i = cand_lookup[prev_s][cand_i_tid]
                    fused, tac_res = scorer.score(cand_i, cand_j, node_i, node_j)
                    if hard_tactic_filter and tac_res.rule == "R9":
                        continue
                    log_trans = math.log(max(fused, _EPS))
                    contribs.append(
                        prev_score
                        + d * log_λ
                        + (1 - α) * cand_j.log_emission
                        + α * log_trans
                    )
            forward[s][cand_j.technique_id] = _logsumexp(contribs) if contribs else _LOG_ZERO

    # ── Backward pass (sum-product) ─────────────────────────────────────────
    backward: list[dict[str, float]] = [{} for _ in range(T)]
    for tid in forward[T - 1]:
        backward[T - 1][tid] = 0.0
    for s in range(T - 2, -1, -1):
        node_i = group_nodes[s]
        for cand_i_tid in forward[s].keys():
            cand_i = cand_lookup[s][cand_i_tid]
            contribs = []
            for d in range(0, min(max_skip, T - 1 - s) + 1):
                next_s = s + 1 + d
                if next_s >= T:
                    break
                node_j = group_nodes[next_s]
                for cand_j_tid, next_score in backward[next_s].items():
                    cand_j = cand_lookup[next_s][cand_j_tid]
                    fused, tac_res = scorer.score(cand_i, cand_j, node_i, node_j)
                    if hard_tactic_filter and tac_res.rule == "R9":
                        continue
                    log_trans = math.log(max(fused, _EPS))
                    contribs.append(
                        next_score
                        + d * log_λ
                        + (1 - α) * cand_j.log_emission
                        + α * log_trans
                    )
            backward[s][cand_i_tid] = _logsumexp(contribs) if contribs else _LOG_ZERO

    # ── Per-step posterior argmax ──────────────────────────────────────────
    best_path: list[Candidate] = []
    total_score = 0.0
    for s in range(T):
        best_tid = None
        best_p = _LOG_ZERO
        for tid, f_val in forward[s].items():
            p = f_val + backward[s].get(tid, _LOG_ZERO)
            if p > best_p:
                best_p = p
                best_tid = tid
        if best_tid is None:
            best_tid = group_nodes[s].candidates[0].technique_id
            cand_lookup[s].setdefault(best_tid, group_nodes[s].candidates[0])
        best_path.append(cand_lookup[s][best_tid])
        total_score = max(total_score, best_p)

    # ── Score breakdown ─────────────────────────────────────────────────────
    breakdown: list[dict] = []
    for s, c in enumerate(best_path):
        entry = {
            "step":           s + 1,
            "group_idx":      s,
            "group_id":       group_nodes[s].group_id,
            "technique_id":   c.technique_id,
            "technique_name": c.technique_name,
            "tactic":         c.tactic,
            "similarity":     c.similarity,
            "log_emission":   round(c.log_emission, 4),
        }
        if s > 0:
            prev = best_path[s - 1]
            fused, tac_res = scorer.score(prev, c, group_nodes[s - 1], group_nodes[s])
            entry["transition_from"]   = prev.tactic
            entry["transition_weight"] = round(fused, 4)
            entry["transition_rule"]   = tac_res.rule
            entry["transition_note"]   = tac_res.note
            entry["log_transition"]    = round(math.log(max(fused, _EPS)), 4)
            entry["skip_distance"]     = 0
        else:
            entry.update({
                "transition_from": None, "transition_weight": None,
                "transition_rule": None, "transition_note": None,
                "log_transition": None, "skip_distance": 0,
            })
        breakdown.append(entry)

    pred_tactics = [c.tactic for c in best_path]
    novelty = 0.0
    closest = ""
    if campaigns:
        novelty, closest = compute_novelty(pred_tactics, campaigns)

    return ViterbiResult(
        groups           = group_nodes,
        best_path        = best_path,
        best_score       = round(total_score, 4),
        score_breakdown  = breakdown,
        novelty_score    = novelty,
        closest_campaign = closest,
    )


def apply_emission_confidence_bypass(
    vit: "ViterbiResult",
    scorer: "MultiDimTransitionScorer",
    sim_threshold: float = 0.70,
) -> "ViterbiResult":
    """X+Z ...: emission top-1 ...sim(top-1) ...

    """
    groups = vit.groups
    path = list(vit.best_path)
    adjusted: list["Candidate"] = []
    for s, c in enumerate(path):
        node = groups[s]
        if not node.candidates:
            adjusted.append(c)
            continue
        emit_top1 = node.candidates[0]
        if emit_top1.similarity >= sim_threshold:
            adjusted.append(emit_top1)
        else:
            adjusted.append(c)

    new_breakdown = []
    for s, c in enumerate(adjusted):
        entry = {
            "step":           s + 1,
            "group_idx":      s,
            "group_id":       groups[s].group_id,
            "technique_id":   c.technique_id,
            "technique_name": c.technique_name,
            "tactic":         c.tactic,
            "similarity":     c.similarity,
            "log_emission":   round(c.log_emission, 4),
        }
        if s > 0:
            prev = adjusted[s - 1]
            fused, tac_res = scorer.score(prev, c, groups[s - 1], groups[s])
            entry["transition_from"]   = prev.tactic
            entry["transition_weight"] = round(fused, 4)
            entry["transition_rule"]   = tac_res.rule
            entry["transition_note"]   = tac_res.note
            entry["log_transition"]    = round(math.log(max(fused, _EPS)), 4)
            entry["skip_distance"]     = 0
        else:
            entry.update({
                "transition_from": None, "transition_weight": None,
                "transition_rule": None, "transition_note": None,
                "log_transition": None, "skip_distance": 0,
            })
        new_breakdown.append(entry)

    return ViterbiResult(
        groups           = groups,
        best_path        = adjusted,
        best_score       = vit.best_score,
        score_breakdown  = new_breakdown,
        novelty_score    = vit.novelty_score,
        closest_campaign = vit.closest_campaign,
    )


def apply_minimum_regret_guard(
    vit: "ViterbiResult",
    scorer: "MultiDimTransitionScorer",
    transition_weight: float,
    margin_threshold: float = 0.0,
) -> "ViterbiResult":
    """Z: Minimum-regret post-hoc guard.


    - Δlog_trans: log P_trans(prev → vit_pick) − log P_trans(prev → emit_top1)
    - Δlog_emit: log_emission(emit_top1) − log_emission(vit_pick)
    - alpha = transition_weight

    """
    α = transition_weight
    groups = vit.groups
    path = list(vit.best_path)
    breakdown = list(vit.score_breakdown)

    if margin_threshold is None:
        return vit

    adjusted = list(path)
    for s in range(len(path)):
        node_s = groups[s]
        emit_top1 = node_s.candidates[0] if node_s.candidates else None
        if emit_top1 is None:
            continue
        vit_pick = adjusted[s]
        if vit_pick.technique_id == emit_top1.technique_id:
            continue

        delta_emit = emit_top1.log_emission - vit_pick.log_emission  # ≥ 0 (top-1 has highest emission)

        if s == 0:
            if delta_emit > margin_threshold:
                adjusted[s] = emit_top1
            continue

        prev = adjusted[s - 1]
        node_prev = groups[s - 1]
        fused_vit, _ = scorer.score(prev, vit_pick, node_prev, node_s)
        fused_emit, _ = scorer.score(prev, emit_top1, node_prev, node_s)
        log_vit = math.log(max(fused_vit, _EPS))
        log_emit_trans = math.log(max(fused_emit, _EPS))
        delta_trans = log_vit - log_emit_trans

        net_gain = α * delta_trans - (1 - α) * delta_emit
        if net_gain <= margin_threshold:
            adjusted[s] = emit_top1

    new_breakdown = []
    for s, c in enumerate(adjusted):
        entry = {
            "step":           s + 1,
            "group_idx":      s,
            "group_id":       groups[s].group_id,
            "technique_id":   c.technique_id,
            "technique_name": c.technique_name,
            "tactic":         c.tactic,
            "similarity":     c.similarity,
            "log_emission":   round(c.log_emission, 4),
        }
        if s > 0:
            prev = adjusted[s - 1]
            fused, tac_res = scorer.score(prev, c, groups[s - 1], groups[s])
            entry["transition_from"]   = prev.tactic
            entry["transition_weight"] = round(fused, 4)
            entry["transition_rule"]   = tac_res.rule
            entry["transition_note"]   = tac_res.note
            entry["log_transition"]    = round(math.log(max(fused, _EPS)), 4)
            entry["skip_distance"]     = 0
        else:
            entry.update({
                "transition_from": None, "transition_weight": None,
                "transition_rule": None, "transition_note": None,
                "log_transition": None, "skip_distance": 0,
            })
        new_breakdown.append(entry)

    return ViterbiResult(
        groups           = groups,
        best_path        = adjusted,
        best_score       = vit.best_score,

        score_breakdown  = new_breakdown,
        novelty_score    = vit.novelty_score,
        closest_campaign = vit.closest_campaign,
    )


# Legacy wrapper
def viterbi_best_path(
    group_nodes: list[GroupNode],
    scorer: TacticalScorer,
    transition_weight: float = 0.5,
) -> ViterbiResult:
    """...API ...-- TacticalScorer..."""
    multi = MultiDimTransitionScorer(tac_scorer=scorer)
    return topk_viterbi(
        group_nodes, multi,
        beam_k=5, max_skip=0, skip_penalty=0.5,
        transition_weight=transition_weight,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def print_viterbi_report(result: ViterbiResult) -> None:
    print("\n" + "═" * 80)
    print("  Top-K Viterbi Attack Chain (with hole-bridging)")
    print("═" * 80)
    print(f"  ...: {len(result.groups)}...")
    print(f"  ...: {len(result.best_path)}...")
    print(f"  ...: {result.best_score:.4f}  (log scale)")
    if result.novelty_score > 0:
        print(f"  Novelty score  : {result.novelty_score:.4f}")
        print(f"  Closest campaign: {result.closest_campaign}")
    print()

    for b in result.score_breakdown:
        s = b["group_idx"]
        t = result.groups[s].anchor_time
        t_str = t.strftime("%H:%M:%S") if t is not None else "??:??:??"

        if b["transition_from"]:
            skip_str = ""
            if b.get("skip_distance", 0) > 0:
                skip_str = f"  [SKIP d={b['skip_distance']}]"
            print(f"  ↑ {b['transition_from']:25s}"
                  f"→ w={b['transition_weight']:.4f}"
                  f" [{b['transition_rule']}] {b['transition_note']}{skip_str}")

        sim_bar = "█" * int(b["similarity"] * 20)
        print(f"  [{b['step']:>2}] {t_str}  {b['group_id']:<22}"
              f"  {b['tactic']:<28}"
              f"  {b['technique_id']:<14}"
              f"  sim={b['similarity']:.4f} {sim_bar}")

    print(f"\n  ── ...weight < 0.1) ──")
    anomalies = [b for b in result.score_breakdown
                 if b["transition_weight"] is not None and b["transition_weight"] < 0.1]
    if anomalies:
        for b in anomalies:
            print(f"  ⚠  {b['transition_from']:25s} → {b['tactic']:25s}"
                  f"  w={b['transition_weight']:.4f}  ({b['transition_rule']})")
    else:
        print("  ✓  ...")

    print("═" * 80)
