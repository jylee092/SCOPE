"""
SHIELD §3 -- LLM Analyzer (3-stage Chain-of-Thought).

Implements Algorithm 1 from Gandhi et al. (SHIELD 2025): for each Louvain
community, run three sequential LLM calls -- (S1) identify suspicious
processes, (S2) per-process behavioral check, (S3) chain analysis →
`(confidence_score, attack_summary, malicious_processes, kill_chain_stages,
tids)`.

Adaptations:
- LLM is Gemini 2.5 Flash (vs. Qwen 2.5 32B in the paper); the API surface
  is wrapped via pipeline.mitre_mapper._call_gemini for consistency with
  SCOPE's existing rate-limit/retry handling.
- Output for S3 is constrained to JSON (response_mime_type) to make
  downstream parsing robust.
- Per-(scenario, community, stage) results are cached on disk so repeated
  runs of the pipeline incur no extra API calls.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, timedelta

import google.generativeai as genai
import networkx as nx
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Gemini call (mirrors pipeline.mitre_mapper._call_gemini, inlined to avoid
# pulling the heavy faiss / sentence-transformers stack just for a helper).
# ---------------------------------------------------------------------------

_GEMINI_TIMEOUT_SEC = 90
_GEMINI_MAX_RETRIES = 5


def _classify_error(e: Exception) -> str:
    s = str(e).lower()
    if any(k in s for k in ("rate", "quota", "429", "exceeded")):
        return "rate_limit"
    if any(k in s for k in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(k in s for k in ("503", "unavailable", "500", "internal")):
        return "server_error"
    return "other"


_RETRY_DELAY_RE = re.compile(r"retry[_ -]delay\D*?(\d+)", re.I)


def _parse_retry_delay(err_text: str) -> Optional[int]:
    m = _RETRY_DELAY_RE.search(err_text or "")
    return int(m.group(1)) if m else None


def _call_gemini(prompt: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    last_err: Optional[Exception] = None
    for attempt in range(1, _GEMINI_MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2500,
                    response_mime_type="application/json",
                ),
                request_options={"timeout": _GEMINI_TIMEOUT_SEC},
            )
            return (response.text or "").strip()
        except Exception as e:
            last_err = e
            kind = _classify_error(e)
            if kind == "rate_limit":
                delay = _parse_retry_delay(str(e)) or min(60 * attempt, 300)
                next_at = datetime.now() + timedelta(seconds=delay)
                print(f"  [SHIELD rate-limit] attempt {attempt}: wait {delay}s "
                      f"(resume ~{next_at.strftime('%H:%M:%S')})")
                time.sleep(delay + 2)
                continue
            if kind == "timeout":
                print(f"  [SHIELD timeout] attempt {attempt}: {e}")
                time.sleep(5 * attempt); continue
            if kind == "server_error":
                print(f"  [SHIELD server-err] attempt {attempt}: {e}")
                time.sleep(10 * attempt); continue
            print(f"  [SHIELD err] {type(e).__name__}: {e}")
            raise
    raise RuntimeError(f"Gemini call failed after {_GEMINI_MAX_RETRIES}: {last_err}")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_DIR = config.OUTPUT_BASE_DIR / "_cache" / "shield"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(scenario: str, comm_id: int, stage: str, payload: str) -> Path:
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return _CACHE_DIR / f"{scenario}__c{comm_id}__{stage}__{h}.json"


def _cached_call(scenario: str, comm_id: int, stage: str, prompt: str,
                 api_key: str, force_json: bool = False) -> str:
    """Wrapper around _call_gemini with on-disk caching."""
    p = _cache_key(scenario, comm_id, stage, prompt)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))["text"]
        except Exception:
            p.unlink(missing_ok=True)
    if force_json:
        # _call_gemini does not expose response_mime_type, so we instead
        # instruct in-prompt and parse defensively. Override here if/when
        # mitre_mapper grows a JSON-mode option.
        pass
    text = _call_gemini(prompt, api_key)
    p.write_text(json.dumps({"text": text}), encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Community → text serialization
# ---------------------------------------------------------------------------

_MAX_NODES = 60                                    # cap per community for context
_MAX_EDGES = 120


def _node_label(g: nx.MultiDiGraph, n: str) -> str:
    a = g.nodes.get(n, {})
    typ = a.get("type", "?")
    if typ == "process":
        img = (a.get("image") or "").rsplit("\\", 1)[-1] or "?"
        guid = (a.get("guid") or "")[:13]
        return f"P[{img}|{guid}]"
    if typ == "file":
        return f"F[{a.get('path', n)[:80]}]"
    if typ == "registry":
        return f"R[{a.get('key', n)[:80]}]"
    if typ == "socket":
        ext = " ext" if a.get("external") else ""
        return f"N[{a.get('host','?')}:{a.get('port','')}{ext}]"
    if typ == "module":
        img = (a.get("image") or "").rsplit("\\", 1)[-1] or "?"
        return f"M[{img}]"
    if typ == "pipe":
        return f"PIPE[{a.get('name', n)[:60]}]"
    return n[:60]


def serialize_community(g: nx.MultiDiGraph, members: list[str],
                         scenario_logs: pd.DataFrame | None = None) -> str:
    """Compact textual representation for LLM input.

    The serialization explicitly surfaces:
      1. Process list (process names)
      2. **Notable command lines** -- distinct fork-edge cmdlines with the
         spawning process; this is the most reliable signal of attack
         tooling, so we promote it to the top of the prompt.
      3. **Notable artifacts** -- file paths, registry keys, pipes, external
         sockets that the community touches.
      4. Chronological event list (truncated).

    Empirically, hiding command lines inside the bottom event list lets the
    LLM Stage-1 miss obvious attacks (e.g., 'esentutl /y /vss', 'rundll32
    comsvcs.dll MiniDump'). Promoting them resolves that without changing
    the underlying SHIELD pipeline.
    """
    sub = g.subgraph(members).copy()
    nodes = list(sub.nodes())[:_MAX_NODES]
    procs = [n for n in nodes if sub.nodes[n].get("type") == "process"]

    lines = []
    lines.append(f"Community: {len(members)} nodes "
                 f"(showing first {len(nodes)}).")
    type_counts = Counter(sub.nodes[n].get("type", "?") for n in nodes)
    lines.append(f"Node types: {dict(type_counts)}")

    if procs:
        lines.append(f"Processes ({len(procs)}):")
        for p in procs[:25]:
            lines.append(f"  - {_node_label(sub, p)}")

    # ---- (2) Notable command lines (deduped) ----
    cmdlines: list[tuple[str, str]] = []  # (process, cmdline)
    cmd_seen: set[str] = set()
    for u, v, d in sub.edges(data=True):
        cli = (d.get("cmdline") or "").strip()
        if not cli or len(cli) < 5:
            continue
        # dedup on cmdline (first 200 chars)
        key = cli[:200]
        if key in cmd_seen:
            continue
        cmd_seen.add(key)
        proc_lbl = _node_label(sub, v) if sub.nodes[v].get("type") == "process" \
                                       else _node_label(sub, u)
        cmdlines.append((proc_lbl, cli[:300]))
    if cmdlines:
        lines.append(f"\nNotable command lines ({len(cmdlines)} distinct):")
        for proc, cli in cmdlines[:20]:
            lines.append(f"  {proc}: {cli}")

    # ---- (3) Notable artifacts ----
    artifacts: dict[str, list[str]] = {"file": [], "registry": [],
                                         "pipe": [], "socket": []}
    for n in nodes:
        attrs = sub.nodes[n]
        t = attrs.get("type")
        if t == "file":
            path = attrs.get("path", n[2:])
            artifacts["file"].append(path)
        elif t == "registry":
            artifacts["registry"].append(attrs.get("key", n[2:]))
        elif t == "pipe":
            artifacts["pipe"].append(attrs.get("name", n[5:]))
        elif t == "socket":
            tag = " (external)" if attrs.get("external") else ""
            artifacts["socket"].append(
                f"{attrs.get('host','?')}:{attrs.get('port','')}{tag}"
            )
    if any(artifacts.values()):
        lines.append(f"\nNotable artifacts:")
        for kind, items in artifacts.items():
            uniq = list(dict.fromkeys(items))[:15]
            if uniq:
                lines.append(f"  {kind}s ({len(items)} total, {len(uniq)} shown):")
                for x in uniq:
                    lines.append(f"    - {x[:160]}")

    # ---- (4) Chronological event list ----
    edges = list(sub.edges(data=True))
    edges.sort(key=lambda e: str(e[2].get("ts") or ""))
    edges = edges[:_MAX_EDGES]
    lines.append(f"\nEvents ({len(edges)} of {sub.number_of_edges()}):")
    for u, v, d in edges:
        verb = d.get("event_type", "?")
        ts = (d.get("ts") or "")[:19]
        cli = d.get("cmdline")
        det = d.get("details")
        extra = f" cli={cli!r}" if cli else (f" details={det!r}" if det else "")
        lines.append(f"  {ts} {_node_label(sub, u)} --[{verb}]--> "
                     f"{_node_label(sub, v)}{extra}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1: identify suspicious processes
# ---------------------------------------------------------------------------

_STAGE1_PROMPT = """\
You are a senior SOC analyst. Inspect the following community of related
system events extracted from a host's provenance graph and identify which
processes are suspicious based on your knowledge of common attack patterns.

Pay particular attention to the **Notable command lines** section: these
are the distinct invocation strings observed in the community. Most
sophisticated attacks are recognizable by their command-line arguments
(e.g., 'rundll32 comsvcs.dll MiniDump' for LSASS dumping, 'esentutl /y
/vss' for SAM hive copy, 'ntdsutil ifm create full' for NTDS.dit dump,
'wevtutil cl' or 'reg add ... MININT' for log tampering, encoded
PowerShell launchers, 'sc config ... binPath' for service mod, 'schtasks
/create' for persistence, etc.) -- even when the parent binary is a
legitimate Windows utility.

Other suspicious indicators (non-exhaustive): unauthorized writes to
sensitive directories (Public, Temp, AppData), unexpected child processes
of cmd/powershell, rare DLL loads (comsvcs, mimikatz components),
registry persistence keys (Run, Services, Image File Execution Options),
external network connections, named-pipe abuse (\\\\.\\pipe\\lsass),
log-clearing, masquerading binaries.

Community:
-----
{community}
-----

Respond with a single JSON object on the form:
{{"suspicious_processes": ["<process label>", ...], "rationale": "<one sentence>"}}
Use the EXACT process labels as they appear in the community description
(e.g., "P[ntdsutil.exe|fe86be69-088]"). Include a process whenever its
command line, child processes, file/registry writes, or network behavior
match a known attack pattern, even if the binary is a legitimate utility.
If truly nothing in the community is suspicious, return an empty list.
"""


def stage1_identify(community_text: str, scenario: str, comm_id: int,
                    api_key: str) -> dict:
    prompt = _STAGE1_PROMPT.format(community=community_text)
    raw = _cached_call(scenario, comm_id, "s1", prompt, api_key)
    return _parse_json(raw, default={"suspicious_processes": [], "rationale": ""})


# ---------------------------------------------------------------------------
# Stage 2: per-process behavioral verification
# ---------------------------------------------------------------------------

_STAGE2_PROMPT = """\
You are checking whether process {process_label} is genuinely suspicious.
Below are the events involving this process, in chronological order.

Events:
-----
{process_events}
-----

Decide:
1. Is this process well-known to you (a typical Windows binary)?
2. If unknown / unusual, mark it suspicious.
3. If known, perform deeper behavioral analysis: does it deviate from
   normal usage (e.g., cmd.exe spawning ntdsutil with IFM args, rundll32
   loading comsvcs.dll for MiniDump, log-clearing wevtutil arguments)?

Respond with a single JSON object:
{{"isSuspicious": <true|false>,
  "deviation": "<short string describing the suspicious deviation, or empty>",
  "indicators": ["<keyword>", ...]}}
"""


def stage2_verify(process_label: str, process_events_text: str,
                  scenario: str, comm_id: int,
                  api_key: str) -> dict:
    prompt = _STAGE2_PROMPT.format(process_label=process_label,
                                    process_events=process_events_text)
    raw = _cached_call(scenario, comm_id, f"s2_{_safe(process_label)}", prompt,
                       api_key)
    return _parse_json(raw, default={"isSuspicious": False,
                                       "deviation": "", "indicators": []})


# ---------------------------------------------------------------------------
# Stage 3: full chain analysis with kill-chain mapping
# ---------------------------------------------------------------------------

_STAGE3_PROMPT = """\
You are a senior SOC analyst producing the final attack summary for one
community. The graph analyzer flagged the following processes as
suspicious, and behavioral verification confirmed deviations:

Confirmed suspicious processes:
{confirmed}

Per-process deviations:
{deviations}

Full community context:
-----
{community}
-----

Tasks:
(a) Decide if these activities cohere into an attack (rather than benign
    coincidences).
(b) Produce a confidence_score in [0, 1]:
       >= 0.9  : a complete, coherent attack chain
       0.8-0.9 : a partial attack chain
       0.7-0.8 : isolated suspicious patterns
       < 0.7   : likely benign
(c) Map the relevant events to MITRE ATT&CK kill-chain stages (one or more
    of: Reconnaissance, Resource Development, Initial Access, Execution,
    Persistence, Privilege Escalation, Defense Evasion, Credential Access,
    Discovery, Lateral Movement, Collection, Command and Control,
    Exfiltration, Impact).
(d) For each identified stage, propose the most likely MITRE technique IDs
    (e.g., T1003.001, T1059.003, T1036.005). Use sub-technique IDs where
    confident; otherwise use the parent (e.g., T1003).

Respond with a single JSON object on the form:
{{
  "confidence_score": <float in [0, 1]>,
  "attack_summary": "<2-4 sentence narrative>",
  "malicious_processes": ["<process label>", ...],
  "kill_chain_stages": ["Execution", "Credential Access", ...],
  "techniques": [
    {{"tactic": "Credential Access", "tid": "T1003.003",
      "evidence": "<one short sentence>"}},
    ...
  ]
}}
"""


def stage3_chain(community_text: str, confirmed: list[str], deviations: dict,
                 scenario: str, comm_id: int, api_key: str) -> dict:
    confirmed_block = "\n".join(f"  - {p}" for p in confirmed) or "  (none)"
    deviations_block = "\n".join(
        f"  - {p}: {d.get('deviation','')}" for p, d in deviations.items()
    ) or "  (none)"
    prompt = _STAGE3_PROMPT.format(
        confirmed=confirmed_block,
        deviations=deviations_block,
        community=community_text,
    )
    raw = _cached_call(scenario, comm_id, "s3", prompt, api_key)
    return _parse_json(raw, default={
        "confidence_score": 0.0, "attack_summary": "",
        "malicious_processes": [], "kill_chain_stages": [], "techniques": [],
    })


# ---------------------------------------------------------------------------
# Per-process events serialization (for stage 2)
# ---------------------------------------------------------------------------

def process_events_text(g: nx.MultiDiGraph, process_node: str,
                          max_edges: int = 50) -> str:
    edges_out = list(g.out_edges(process_node, data=True))
    edges_in = list(g.in_edges(process_node, data=True))
    edges = sorted(edges_out + edges_in,
                   key=lambda e: str(e[2].get("ts") or ""))[:max_edges]
    if not edges:
        return "(no events for this process)"
    lines = []
    for u, v, d in edges:
        verb = d.get("event_type", "?")
        ts = (d.get("ts") or "")[:19]
        cli = d.get("cmdline")
        det = d.get("details")
        extra = f" cli={cli!r}" if cli else (f" details={det!r}" if det else "")
        lines.append(f"  {ts} {_node_label(g, u)} --[{verb}]--> "
                     f"{_node_label(g, v)}{extra}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level community analysis (Algorithm 1)
# ---------------------------------------------------------------------------

@dataclass
class CommunityAnalysis:
    community_id: int
    n_members: int
    suspicious_processes: list[str] = field(default_factory=list)
    deviations: dict[str, dict] = field(default_factory=dict)
    chain: dict = field(default_factory=dict)
    confidence: float = 0.0
    techniques: list[dict] = field(default_factory=list)
    kill_chain_stages: list[str] = field(default_factory=list)
    summary: str = ""
    skipped: bool = False
    skip_reason: str = ""


def analyze_community(g: nx.MultiDiGraph,
                       comm_id: int,
                       members: list[str],
                       scenario: str,
                       api_key: str,
                       min_size: int = 3,
                       confidence_threshold: float = 0.7,
                       ) -> CommunityAnalysis:
    """Run Algorithm 1 on a single community."""
    res = CommunityAnalysis(community_id=comm_id, n_members=len(members))

    if len(members) < min_size:
        res.skipped = True
        res.skip_reason = f"too small ({len(members)} < {min_size})"
        return res

    # Build community text
    community_text = serialize_community(g, members)

    # ---- Stage 1: identify suspicious processes ----
    s1 = stage1_identify(community_text, scenario, comm_id, api_key)
    sus = [p for p in (s1.get("suspicious_processes") or []) if p]
    res.suspicious_processes = sus
    if not sus:
        res.skipped = True
        res.skip_reason = "no suspicious processes found in S1"
        return res

    # Map LLM-emitted labels back to actual nodes
    label_to_node = {_node_label(g, n): n for n in members
                     if g.nodes[n].get("type") == "process"}
    matched = []
    for label in sus:
        node = label_to_node.get(label)
        if not node:
            # fuzzy match: process containing the same image name
            img = re.search(r"P\[([^|]+)", label)
            if img:
                want = img.group(1).lower()
                for lbl, n in label_to_node.items():
                    if want in lbl.lower():
                        node = n
                        break
        if node:
            matched.append((label, node))

    if not matched:
        res.skipped = True
        res.skip_reason = "S1 labels did not resolve to nodes"
        return res

    # ---- Stage 2: per-process verification ----
    deviations: dict[str, dict] = {}
    for label, node in matched[:5]:                 # cap at 5 to control cost
        ev_text = process_events_text(g, node)
        s2 = stage2_verify(label, ev_text, scenario, comm_id, api_key)
        if s2.get("isSuspicious"):
            deviations[label] = s2
    res.deviations = deviations
    if not deviations:
        res.skipped = True
        res.skip_reason = "S2 found no real deviations"
        return res

    # ---- Stage 3: chain analysis ----
    s3 = stage3_chain(community_text, [lbl for lbl, _ in matched],
                       deviations, scenario, comm_id, api_key)
    res.chain = s3
    res.confidence = float(s3.get("confidence_score") or 0.0)
    res.techniques = s3.get("techniques") or []
    res.kill_chain_stages = s3.get("kill_chain_stages") or []
    res.summary = s3.get("attack_summary") or ""

    if res.confidence < confidence_threshold:
        # Below δ -- SHIELD only "tags + traces" without raising alert.
        # We still keep the result for downstream chain assembly because we
        # want to recover *as much* attack signal as possible across the whole
        # scenario, mirroring SHIELD's reduced-graph attack set T.
        pass
    return res


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")


def _safe(label: str) -> str:
    return _SAFE_RE.sub("_", label)[:40]


def _parse_json(text: str, default: dict | None = None) -> dict:
    """Robust JSON extraction -- handles ```json``` fences and trailing prose."""
    if not text:
        return dict(default or {})
    s = text.strip()
    # Strip markdown code fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Find first {...} block
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return dict(default or {})
    body = m.group(0)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Try to fix common issues: trailing commas, single quotes
        fixed = re.sub(r",(\s*[}\]])", r"\1", body)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return dict(default or {})
