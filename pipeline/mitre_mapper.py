"""


--------
analyze(all_features, mitre_csv_path, gemini_api_key) -> list[dict]
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import faiss
import google.generativeai as genai
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import EMBED_MODEL_NAME, GEMINI_MODEL, TOP_K

SOFTMAX_BETA = 10.0

# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
_EMBED_MODEL: Optional[SentenceTransformer] = None
_INDEX_CACHE: dict = {}   # key: (csv_path, csv_hash) → (index, meta_df)


def _file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_embed_model() -> SentenceTransformer:
    """...SentenceTransformer ..."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print(f"  [cache-miss] SentenceTransformer ...: {EMBED_MODEL_NAME}")
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
_LLM_CACHE: Optional[dict] = None
_LLM_CACHE_PATH: Optional[Path] = None


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _load_llm_cache(cache_dir: Path) -> dict:
    """...LLM ...dict ..."""
    global _LLM_CACHE, _LLM_CACHE_PATH
    _LLM_CACHE_PATH = Path(cache_dir) / "llm_descriptions.json"
    if _LLM_CACHE is not None:
        return _LLM_CACHE

    if _LLM_CACHE_PATH.exists():
        try:
            with open(_LLM_CACHE_PATH, encoding="utf-8") as f:
                _LLM_CACHE = json.load(f)
            print(f"  [cache-hit] LLM ...: {len(_LLM_CACHE)}...")
        except (json.JSONDecodeError, OSError):
            print(f"  [cache-warn] LLM ...")
            _LLM_CACHE = {}
    else:
        _LLM_CACHE = {}
    return _LLM_CACHE


def _save_llm_cache() -> None:
    if _LLM_CACHE is None or _LLM_CACHE_PATH is None:
        return
    _LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _LLM_CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_LLM_CACHE, f, ensure_ascii=False, indent=2)
    tmp.replace(_LLM_CACHE_PATH)


_GEMINI_TIMEOUT_SEC = 60
_GEMINI_MAX_RETRIES = 6


def _parse_retry_delay(err_text: str) -> Optional[int]:
    """Gemini 429 ...retry_delay ..."""
    if "retry" in err_text.lower():
        m = re.search(r"seconds[\"']?\s*[:=]\s*(\d+)", err_text)
        if m:
            return int(m.group(1))
    m = re.search(r"Retry-After:\s*(\d+)", err_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _classify_error(e: Exception) -> str:
    s = str(e).lower()
    if any(k in s for k in ("429", "resource_exhausted", "rate limit", "quota", "exhausted")):
        return "rate_limit"
    if any(k in s for k in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(k in s for k in ("503", "unavailable", "500", "internal")):
        return "server_error"
    return "other"


def _call_gemini(prompt: str, api_key: str) -> str:
    """...+ retry ...Gemini ...rate limit ...retry_delay ..."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    last_err: Optional[Exception] = None
    for attempt in range(1, _GEMINI_MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.35,

                    max_output_tokens=2500,
                ),
                request_options={"timeout": _GEMINI_TIMEOUT_SEC},
            )
            return response.text.strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            kind = _classify_error(e)

            if kind == "rate_limit":
                delay = _parse_retry_delay(str(e))
                if delay is None:
                    delay = min(60 * attempt, 300)  # exponential backoff fallback
                next_at = datetime.now() + timedelta(seconds=delay)
                print(f"  [rate-limit] ...{attempt}/{_GEMINI_MAX_RETRIES}: "
                      f"{delay}...≈ {next_at.strftime('%H:%M:%S')}")
                time.sleep(delay + 2)
                continue

            if kind == "timeout":
                print(f"  [timeout] ...{attempt}/{_GEMINI_MAX_RETRIES}: {e}")
                time.sleep(5 * attempt)
                continue

            if kind == "server_error":
                print(f"  [server-err] ...{attempt}/{_GEMINI_MAX_RETRIES}: {e}")
                time.sleep(10 * attempt)
                continue

            print(f"  [err] {type(e).__name__}: {e}")
            raise

    raise RuntimeError(f"Gemini ...{_GEMINI_MAX_RETRIES}...: {last_err}")


def generate_description_cached(
    feat: dict,
    api_key: str,
    cache_dir: Optional[Path] = None,
) -> tuple[str, bool]:
    """...miss ...Gemini ...Returns (desc, was_cached)."""
    prompt = build_prompt(feat)
    key = _prompt_hash(prompt)

    if cache_dir is not None:
        cache = _load_llm_cache(Path(cache_dir))
        entry = cache.get(key)
        if entry is not None and entry.get("model") == GEMINI_MODEL:
            return entry["description"], True

    desc = _call_gemini(prompt, api_key)

    if cache_dir is not None:
        cache[key] = {
            "description": desc,
            "model": GEMINI_MODEL,
            "prompt_preview": prompt[:200],
        }
        _save_llm_cache()

    return desc, False


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def clean_mitre_description(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = re.sub(r"<[^>]+>", "", text)
    t = re.sub(r"\(Citation:[^\)]+\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_chains(chains: list) -> str:
    if not chains:
        return "  (none)"
    lines = []
    for c in chains:
        parent = c.get("parent_image") or "?"
        child  = c.get("child_image")  or "?"
        if c.get("relation", "spawn") == "access":
            access = c.get("granted_access", "")
            lines.append(f"  {parent} → accessed memory of {child}"
                         + (f" (access_mask={access})" if access else ""))
        else:
            lines.append(f"  {parent} → spawned {child}")
    return "\n".join(lines)


def _fmt_cmdlines(entries: list) -> str:
    if not entries:
        return "  (none)"
    lines = []
    for e in entries:
        img = e.get("image") or "?"
        cl  = e.get("cmdline") or ""
        obf = " [OBFUSCATED]" if e.get("has_obfuscation") else ""
        lines.append(f"  [{img}]{obf}  {cl}")
    return "\n".join(lines)


def _fmt_registry(signals: list) -> str:
    if not signals:
        return "  (none)"
    return "\n".join(
        f"  {s.get('target_object') or ''}  →  {s.get('details') or '(empty)'}"
        for s in signals
    )


def _fmt_files(files: list) -> str:
    if not files:
        return "  (none)"
    return "\n".join(
        f"  [{f.get('image','?')}] → {f.get('path','')}" for f in files
    )


def _fmt_network(connections: list) -> str:
    if not connections:
        return "  (none)"
    lines = []
    for c in connections:
        direction = c.get("direction", "outbound")
        img  = c.get("image") or "unknown"
        hint = f" [{c['service_hint']}]" if c.get("service_hint") else ""
        if direction == "listen":
            port = c.get("listen_port") or c.get("src_port", "")
            lines.append(f"  LISTEN   port={port}  process={img}{hint}")
        elif direction == "dns_query":
            lines.append(f"  DNS      query={c.get('query_name') or ''}  process={img}")
        else:
            dst_ip   = c.get("dst_ip")   or ""
            dst_port = c.get("dst_port") or ""
            src_ip   = c.get("src_ip")   or ""
            proto    = c.get("protocol") or ""
            lines.append(f"  OUTBOUND {src_ip} → {dst_ip}:{dst_port}  {proto}  process={img}{hint}")
    return "\n".join(lines)


def _fmt_evasion(evasion: dict) -> str:
    parts = []
    if evasion.get("log_cleared"):
        eids = ", ".join(str(e["source_eid"]) for e in evasion["log_cleared"])
        parts.append(f"Event logs cleared (EID {eids})")
    deleted = evasion.get("deleted_files") or []
    if deleted:
        parts.append(f"Files deleted ({len(deleted)}):")
        for d in deleted[:8]:
            parts.append(f"    [{d.get('image','?')}] → {d.get('path','')}")
    obf = evasion.get("obfuscated_cmdlines") or []
    if obf:
        parts.append(f"Obfuscated command lines ({len(obf)}):")
        for o in obf[:5]:
            parts.append(f"    [{o.get('image','?')}] {(o.get('cmdline') or '')[:150]}")
    return "\n".join(f"  {p}" for p in parts) if parts else "  (none)"


def _fmt_identity(idn: dict) -> str:
    if not idn:
        return "  (none)"
    lines = []
    user = idn.get("user")
    domain = idn.get("domain")
    if user:
        user_str = (domain + "\\" + user) if domain else user
        lines.append(f"User: {user_str}")
    integrity = idn.get("integrity_level")
    if integrity:
        lines.append(f"Integrity level: {integrity}")
    logon = idn.get("logon_id")
    if logon:
        lines.append(f"Logon ID: {logon}")
    privs = idn.get("privilege_list")
    if privs:
        lines.append(f"Privileges: {str(privs)[:200]}")
    sid = idn.get("user_sid")
    if sid:
        lines.append(f"User SID: {sid}")
    if not lines:
        return "  (unknown)"
    return "\n".join(f"  {l}" for l in lines)


def _fmt_temporal_extended(tmp: dict) -> str:
    """Extended temporal info including EID distribution."""
    span    = abs(tmp.get("window_end_delta_sec", 0) - tmp.get("window_start_delta_sec", 0))
    total   = tmp.get("total_events", 0)
    density = tmp.get("event_density_per_10sec")
    bursts  = tmp.get("burst_detected", [])
    eid_counts = tmp.get("eid_counts") or {}

    lines = [f"  {total} events over {span:.1f} seconds"]
    if density:
        lines.append(f"  Event density: {density} events/10sec")
    if eid_counts:
        top_eids = sorted(eid_counts.items(), key=lambda x: -x[1])[:6]
        lines.append(f"  EID distribution: " + ", ".join(f"EID {e} ×{c}" for e, c in top_eids))
    if bursts:
        burst_str = ", ".join(f"EID {b['eid']} x{b['count']} in {b['window_sec']}s" for b in bursts)
        lines.append(f"  Burst detected: {burst_str}")
    return "\n".join(lines)


def _fmt_temporal(tmp: dict) -> str:
    span    = abs(tmp.get("window_end_delta_sec", 0) - tmp.get("window_start_delta_sec", 0))
    total   = tmp.get("total_events", 0)
    density = tmp.get("event_density_per_10sec")
    bursts  = tmp.get("burst_detected", [])

    lines = [f"  {total} events over {span:.1f} seconds"]
    if density:
        lines.append(f"  Event density: {density} events/10sec")
    if bursts:
        burst_str = ", ".join(f"EID {b['eid']} x{b['count']} in {b['window_sec']}s" for b in bursts)
        lines.append(f"  Burst detected: {burst_str}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_anchor_detail(anchor: dict, rule_tid: str, rule_name: str) -> str:
    """Rule...anchor event...LLM..."""
    if not anchor:
        return f"  (rule {rule_tid} {rule_name}: anchor detail unavailable)"
    lines = [f"  Triggered by rule: {rule_tid} ({rule_name})"]
    if anchor.get("eid") is not None:
        lines.append(f"  Anchor EventID: {anchor['eid']}")
    if anchor.get("image"):
        parent = anchor.get("parent_image")
        if parent:
            lines.append(f"  Image: {anchor['image']} (parent: {parent})")
        else:
            lines.append(f"  Image: {anchor['image']}")
    if anchor.get("cmdline"):
        lines.append(f"  CommandLine: {anchor['cmdline']}")
    if anchor.get("target_image"):
        lines.append(f"  Target image: {anchor['target_image']}"
                     + (f" (GrantedAccess={anchor['granted_access']})" if anchor.get("granted_access") else ""))
    if anchor.get("target_object"):
        lines.append(f"  Target registry/object: {anchor['target_object']}")
    if anchor.get("target_file"):
        lines.append(f"  Target file: {anchor['target_file']}")
    if anchor.get("query_name"):
        lines.append(f"  DNS query: {anchor['query_name']}")
    if anchor.get("dst_ip") or anchor.get("dst_port"):
        lines.append(f"  Destination: {anchor.get('dst_ip','?')}:{anchor.get('dst_port','?')}")
    if anchor.get("integrity"):
        lines.append(f"  Integrity: {anchor['integrity']}")
    return "\n".join(lines)


def build_prompt(feat: dict) -> str:
    """feature dict → Chain-of-Thought ..."""
    f     = feat["features"]
    per = f.get("persistence")       or {}
    eva = f.get("evasion")           or {}
    cmd = f.get("command_script")    or {}
    ctx = f.get("execution_context") or {}
    net = f.get("network")           or {}
    tmp = f.get("temporal")          or {}
    idn = f.get("identity")          or {}
    anchor = feat.get("anchor_detail") or {}
    rule_tid  = feat.get("technique_id", "")
    rule_name = feat.get("technique_name", "")

    signals = []
    has_lsass_access = any(
        "lsass" in (c.get("child_image") or "").lower()
        for c in ctx["process_chains"] if c.get("relation") == "access"
    )
    has_log_clear    = bool(eva.get("log_cleared"))
    has_obfuscation  = cmd.get("has_obfuscation", False)
    has_network      = bool(net.get("connections"))
    has_persistence  = bool(per.get("registry_signals"))
    has_deleted      = bool(eva.get("deleted_files"))
    has_dump_file    = any(
        ".dmp" in (f_.get("path") or "").lower()
        for f_ in per.get("dropped_files", [])
    )
    # Additional signals
    reg_signals = per.get("registry_signals") or []

    def _is_vss_internal(target: str) -> bool:
        """VSS snapshot internal logging writes -- not a real persistence/service install."""
        t = (target or "").lower()
        return ("\\vss\\diag\\" in t) or ("vssapipublisher" in t) or ("eserecoverywriter" in t)

    has_run_key      = any("run" in (s.get("target_object") or "").lower() for s in reg_signals)
    has_uac_hijack   = any("shell\\open\\command" in (s.get("target_object") or "").lower() for s in reg_signals)
    has_service_reg  = any(
        "\\services\\" in (s.get("target_object") or "").lower()
        and not _is_vss_internal(s.get("target_object"))
        for s in reg_signals
    )

    # Credential-dump cmdline cues. The cmdline is the most diagnostic feature for
    # several T1003 sub-techniques (SAM/SECURITY/SYSTEM/NTDS hive copy, often via
    # esentutl /vss or vssadmin/wbadmin), but the LLM tends to summarize them away
    # unless we surface them as an explicit signal.
    _HIVE_PATHS = (
        "config\\sam", "config/sam",
        "config\\security", "config/security",
        "config\\system", "config/system",
        "ntds.dit", "system32\\config", "system32/config",
    )
    _HIVE_TOOLS = ("ntdsutil", "esentutl", "vssadmin", "wbadmin", "diskshadow")
    has_hive_path = False
    has_hive_tool = False
    for _e in (cmd.get("entries") or []):
        _cl = (_e.get("cmdline") or "").lower()
        if any(p in _cl for p in _HIVE_PATHS):
            has_hive_path = True
        if any(t in _cl for t in _HIVE_TOOLS):
            has_hive_tool = True

    # NTDS.dit / SAM hive dropped artifact (e.g., esentutl /d C:\ProgramData\SAM)
    has_hive_drop = any(
        any(name in (f_.get("path") or "").lower()
            for name in ("\\sam", "/sam", "ntds.dit", "\\security", "/security", "\\system", "/system"))
        and not (f_.get("path") or "").lower().endswith(".log")
        for f_ in per.get("dropped_files", [])
    )
    # Privileged context
    integrity = (idn.get("integrity_level") or "").lower()
    is_elevated = integrity in ("high", "system")
    user_name = (idn.get("user") or "").lower()
    is_system_account = any(k in user_name for k in ("system", "local service", "network service"))

    if has_log_clear:    signals.append("Event log clearing detected -- likely evidence destruction.")
    if has_lsass_access: signals.append("Direct memory access to LSASS -- credential dumping behavior.")
    if has_dump_file:    signals.append("Memory dump file created -- process memory extraction.")
    if has_obfuscation:  signals.append("Obfuscated/encoded command line detected.")
    # Credential-hive signals take precedence -- they pinpoint specific T1003 sub-techniques.
    if has_hive_path:
        signals.append(
            "Registry hive file referenced in cmdline (SAM/SECURITY/SYSTEM/NTDS) -- "
            "credential database extraction in progress."
        )
    if has_hive_tool and not has_hive_path:
        signals.append(
            "Hive/snapshot utility invoked (ntdsutil/esentutl/vssadmin/wbadmin) -- "
            "likely Volume Shadow Copy or database extraction step."
        )
    if has_hive_drop:
        signals.append("Copied registry-hive file dropped to disk -- staged credential database.")
    if has_run_key:      signals.append("Autostart registry key modification -- persistence via Run/RunOnce.")
    if has_uac_hijack:   signals.append("Shell\\Open\\Command registry modified -- possible UAC bypass via hijacked handler.")
    if has_service_reg:  signals.append("Service registry modified -- possible service install/hijack.")
    if has_persistence and not (has_run_key or has_uac_hijack or has_service_reg):
        signals.append("Suspicious registry modification detected.")
    if has_deleted:      signals.append("Files deleted during activity -- possible indicator wipe.")
    if has_network:      signals.append("Network activity observed -- possible C2 or exfiltration.")
    if is_elevated and not is_system_account:
        signals.append(f"Activity running at elevated integrity ({integrity}) under non-system account.")

    signal_block = (
        "\n".join(f"  ⚠ {s}" for s in signals)
        if signals else "  (no strong signals auto-detected)"
    )

    return f"""You are a cybersecurity threat analyst.
Analyze the forensic evidence below and produce a MITRE ATT&CK-style behavior
description that can be semantically matched against official ATT&CK technique
descriptions via sentence-embedding similarity.

---

## Seed hint (from rule matcher; MAY BE WRONG)
A lightweight rule flagged this activity as possibly related to
**{rule_tid} -- {rule_name}**. Use this only as a weak orientation clue. If the
evidence suggests a different sub-technique (e.g., the rule says T1059 but the
command shows clear PowerShell usage -- a T1059.001 signature), describe what
the evidence actually shows.

## Anchor event (what triggered the rule)
{_fmt_anchor_detail(anchor, rule_tid, rule_name)}

---

## Detected Signals
{signal_block}

---

## Forensic Evidence

### Process Execution Chain
{_fmt_chains(ctx['process_chains'])}

### Command Lines Observed
{_fmt_cmdlines(cmd['entries'])}

### Registry Modifications (suspicious paths only)
{_fmt_registry(per['registry_signals'])}

### Files Created
{_fmt_files(per['dropped_files'])}

### Network Activity
{_fmt_network(net['connections'])}

### Evasion Activity
{_fmt_evasion(eva)}

### Timing
{_fmt_temporal_extended(tmp)}

### Identity / Privilege Context
{_fmt_identity(idn)}

---

## Writing Style -- match MITRE ATT&CK in abstraction, NOT in content

Your description should MIRROR the abstraction level of ATT&CK technique text
but MUST be GROUNDED in the evidence above. Do NOT copy any sentence from the
stylistic guide below; these are STYLE cues, not answer templates.

Style guide (abstraction patterns only -- do not quote):
- Begin with "Adversaries ..." or "An adversary ...".
- Describe the OPERATIONAL ACTION (what mechanism the adversary is invoking)
  and the TACTICAL GOAL (why), at roughly the level of an ATT&CK description.
- Name a specific Windows component, OS subsystem, protocol, or binary ONLY
  when it is DIAGNOSTIC for distinguishing the technique from its siblings
  (e.g., PowerShell vs. command shell vs. VBScript; LSASS vs. NTDS vs. SAM
  hive; WinRM vs. SMB vs. DCOM; Run key vs. Scheduled Task vs. WMI
  subscription). Irrelevant helper processes, unrelated browsers, or generic
  infrastructure must stay generic.
- Prefer generic artifact types ("a memory dump file", "an autostart registry
  key", "a scheduled task", "an obfuscated command line") over per-host
  values (absolute paths, user SIDs, IP addresses, specific filenames).
- If obfuscation/encoding was detected in the evidence, use "obfuscated" or
  "encoded".
- If the evidence does not support the seed hint, describe what the evidence
  actually shows.

## Hard constraints

1. BASE your description on the EVIDENCE, not on any memorized ATT&CK text.
   The Command Lines block is the single most diagnostic feature; treat any
   ATT&CK-relevant artifact appearing in a cmdline as REQUIRED content for
   your output, NOT as detail to summarize away. Specifically, if any cmdline
   names `ntdsutil`, `ntds.dit`, `esentutl`, the `SAM`/`SYSTEM`/`SECURITY`
   registry hive (whether by hive name or by full path under
   `system32\config\`), `comsvcs.dll`, `procdump`, `mimikatz`, `rundll32`,
   `reg add`, `schtasks`, `wevtutil`, `sc config`, `cmstp`, `fodhelper`,
   `InprocServer32`, `vssadmin`/`wbadmin`/`diskshadow`, or a `/vss` flag,
   the description MUST mention BOTH the tool/binary AND its target artifact
   (e.g., "esentutl invoked with /vss to copy the SAM hive"). If a cmdline
   targets a credential-store file or hive, the description MUST identify the
   action as credential-database extraction rather than generic registry or
   utility execution. Conversely, if the evidence does NOT mention such an
   artifact, DO NOT invent one.
2. Do NOT quote or paraphrase phrases longer than 6 consecutive words from
   ATT&CK descriptions you may have memorized. Write in your own words based
   on the observed evidence.
3. 2–4 sentences. Minimum 50 words, maximum 180 words.
4. Plain text only. No bullets, no markdown, no code fences, no section labels.

Write ONLY the description (no preamble, no "Description:" prefix).
"""


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def generate_description(feat: dict, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        build_prompt(feat),
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2500,

        ),
    )
    return response.text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def build_faiss_index(mitre_csv_path: str, embed_model: SentenceTransformer):
    """MITRE CSV → IndexFlatIP + ...DataFrame."""
    df = pd.read_csv(mitre_csv_path)

    col_map: dict = {}
    for c in ["technique_id", "ID", "id", "TechniqueID"]:
        if c in df.columns: col_map["id"] = c; break
    for c in ["technique_name", "name", "Name", "TechniqueName"]:
        if c in df.columns: col_map["name"] = c; break
    for c in ["description", "Description", "desc"]:
        if c in df.columns: col_map["desc"] = c; break

    if "desc" not in col_map:
        raise ValueError(f"description ...: {list(df.columns)}")

    desc_col = col_map["desc"]
    meta_df = df[df[desc_col].notna()].copy().reset_index(drop=True)
    meta_df["_desc_clean"] = meta_df[desc_col].apply(clean_mitre_description)
    meta_df = meta_df[meta_df["_desc_clean"].str.len() > 10].reset_index(drop=True)

    rename = {col_map["desc"]: "_desc"}
    if "id" in col_map:   rename[col_map["id"]]   = "_tid"
    if "name" in col_map: rename[col_map["name"]] = "_tname"
    meta_df = meta_df.rename(columns=rename)

    tactic_col = next((c for c in meta_df.columns if c.lower() == "tactics"), None)
    if tactic_col and tactic_col != "_tactics":
        meta_df["_tactics"] = meta_df[tactic_col].fillna("").astype(str)
    elif "_tactics" not in meta_df.columns:
        meta_df["_tactics"] = ""

    if "_tname" in meta_df.columns:
        meta_df["_embed_text"] = meta_df["_tname"].fillna("") + ": " + meta_df["_desc_clean"]
    else:
        meta_df["_embed_text"] = meta_df["_desc_clean"]

    def _mutable_text(cell):
        if not isinstance(cell, str) or not cell.strip():
            return ""
        try:
            items = json.loads(cell)
        except Exception:
            return cell
        if not isinstance(items, list):
            return ""
        parts: list[str] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            field = str(it.get("Field", "")).strip()
            desc  = str(it.get("Description", "")).strip()
            if field:
                parts.append(field)
            if desc:
                parts.append(desc)
        return " ".join(parts)

    extras: list[pd.Series] = [meta_df["_embed_text"]]
    for col in ["Detection Name", "Analysis Description"]:
        if col in meta_df.columns:
            extras.append(meta_df[col].fillna("").astype(str))
    if "Mutable Elements" in meta_df.columns:
        extras.append(meta_df["Mutable Elements"].fillna("").map(_mutable_text))
    meta_df["_bm25_text"] = extras[0]
    for s in extras[1:]:
        meta_df["_bm25_text"] = meta_df["_bm25_text"] + " " + s

    texts = meta_df["_embed_text"].tolist()
    print(f"  MITRE CSV: {len(df)}...{len(meta_df)}...")
    print(f"  ...{EMBED_MODEL_NAME})")

    embeddings = embed_model.encode(
        texts, batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    print(f"  FAISS ...: {index.ntotal}...dim={embeddings.shape[1]}")
    return index, meta_df


_FAISS_SCHEMA_VERSION = "v3"

_BM25_CACHE: dict = {}  # (csv_path, hash) -> (BM25Okapi, tokens_list)
_TID_SIGNATURES: Optional[dict[str, list[str]]] = None  # lazy-loaded


def _load_tid_signatures() -> dict[str, list[str]]:
    """Technique Rule ...TID → signature ...1..."""
    global _TID_SIGNATURES
    if _TID_SIGNATURES is not None:
        return _TID_SIGNATURES
    from pathlib import Path as _P
    here = _P(__file__).resolve().parent.parent
    fp = here / "TTP_Data" / "tid_signatures.json"
    if fp.exists():
        with open(fp, encoding="utf-8") as f:
            _TID_SIGNATURES = json.load(f)
        print(f"  [signature] {len(_TID_SIGNATURES)} TIDs loaded from {fp.name}")
    else:
        _TID_SIGNATURES = {}
        print(f"  [signature] {fp} ...signature rerank ...")
    return _TID_SIGNATURES


def _signature_match_score(description: str, tid: str, sigs_by_tid: dict) -> float:
    """description ...TID signature ...normalize.

    """
    sigs = sigs_by_tid.get(tid) or []
    if not sigs:
        base = tid.split(".")[0]
        sigs = sigs_by_tid.get(base) or []
        if not sigs:
            return 0.0
    dl = description.lower()
    matches = 0
    for s in sigs:
        if s and s in dl:
            matches += 1
    if matches == 0:
        return 0.0
    return min(0.4 + 0.2 * (matches - 1), 1.0)


def _tokenize(text: str) -> list[str]:
    """BM25...-- ...CamelCase, ..."""
    if not text:
        return []
    t = text.lower()
    t = re.sub(r"[^a-z0-9._\\\\/]", " ", t)
    toks = [w.strip(".") for w in t.split() if w.strip(".") and len(w.strip(".")) >= 2]
    return toks


def _build_bm25(meta_df: pd.DataFrame) -> BM25Okapi:
    """MITRE corpus...BM25 ...

    """
    src_col = "_bm25_text" if "_bm25_text" in meta_df.columns else "_embed_text"
    texts = meta_df[src_col].tolist()
    tokens = [_tokenize(t) for t in texts]
    return BM25Okapi(tokens)


def _get_bm25(meta_df: pd.DataFrame, csv_hash: str) -> BM25Okapi:
    key = (id(meta_df), csv_hash)
    if key not in _BM25_CACHE:
        _BM25_CACHE[key] = _build_bm25(meta_df)
    return _BM25_CACHE[key]


def build_or_load_faiss_index(
    mitre_csv_path: str | Path,
    embed_model: SentenceTransformer,
    cache_dir: Optional[str | Path] = None,
):
    """...FAISS ...

    """
    csv_hash = _file_hash(mitre_csv_path)
    model_tag = EMBED_MODEL_NAME.replace("/", "_").replace(":", "_")
    cache_tag = f"{csv_hash}_{_FAISS_SCHEMA_VERSION}_{model_tag}"
    mem_key = (str(Path(mitre_csv_path).resolve()), cache_tag)

    if mem_key in _INDEX_CACHE:
        print(f"  [cache-hit] FAISS ...{cache_tag})")
        return _INDEX_CACHE[mem_key]

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        idx_path = cache_dir / f"mitre_{cache_tag}.index"
        meta_path = cache_dir / f"mitre_{cache_tag}_meta.pkl"
        if idx_path.exists() and meta_path.exists():
            print(f"  [cache-hit] FAISS ...: {idx_path.name}")
            index = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                meta_df = pickle.load(f)
            _INDEX_CACHE[mem_key] = (index, meta_df)
            return index, meta_df

    print(f"  [cache-miss] FAISS ...schema={_FAISS_SCHEMA_VERSION})")
    index, meta_df = build_faiss_index(mitre_csv_path, embed_model)
    _INDEX_CACHE[mem_key] = (index, meta_df)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        idx_path = cache_dir / f"mitre_{cache_tag}.index"
        meta_path = cache_dir / f"mitre_{cache_tag}_meta.pkl"
        faiss.write_index(index, str(idx_path))
        with open(meta_path, "wb") as f:
            pickle.dump(meta_df, f)
        print(f"  [cache-write] FAISS ...: {idx_path.name}")

    return index, meta_df


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def _rule_tactic_of(rule_tid: str, meta_df: pd.DataFrame) -> str:
    """Rule...technique_id...tactic...sub-technique...parent..."""
    if not rule_tid or "_tactics" not in meta_df.columns:
        return ""
    match = meta_df[meta_df.get("_tid", "") == rule_tid]
    if not match.empty:
        raw = str(match.iloc[0].get("_tactics", "")).strip()
        if raw and raw != "nan":
            return raw.split(",")[0].strip()
    if "." in rule_tid:
        return _rule_tactic_of(rule_tid.split(".")[0], meta_df)
    return ""


def search_similar(description: str, index, meta_df: pd.DataFrame,
                   embed_model: SentenceTransformer, k: int = TOP_K,
                   beta: float = SOFTMAX_BETA,
                   rule_tid: str = "",
                   tid_prior: float = 1.15,

                   tactic_prior: float = 1.05,
                   cross_encoder=None,
                   ce_rerank_width: int = 20,
                   ce_weight: float = 0.6,
                   bm25_weight: float = 0.3,
                   bm25_rerank_width: int = 30,
                   signature_weight: float = 0.0,
                   signature_rerank_width: int = 10,
                   family_boost: float = 0.0,
                   family_boost_width: int = 10) -> tuple[list, float]:
    """Top-K ...technique ...+ softmax P_ttp ...Eq. 2).



    Returns:
        (candidates, confidence_margin)  where margin = P(t^1) - P(t^2)
    """
    vec = embed_model.encode(
        [description],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    all_scores, all_indices = index.search(vec, index.ntotal)
    all_sims = all_scores[0]

    # softmax: P_ttp(t|b) = exp(β·sim) / Σ exp(β·sim)
    scaled = beta * all_sims
    scaled -= scaled.max()
    exp_scaled = np.exp(scaled)

    # ── Rule family prior ──────────────────────────────────────────────────
    rule_tactic = _rule_tactic_of(rule_tid, meta_df) if rule_tid else ""
    rule_parent = rule_tid.split(".")[0] if rule_tid else ""
    if rule_tid and (tid_prior != 1.0 or tactic_prior != 1.0):
        for pos, orig_idx in enumerate(all_indices[0]):
            row = meta_df.iloc[orig_idx]
            cand_tid = str(row.get("_tid", ""))
            cand_parent = cand_tid.split(".")[0]
            if rule_parent and cand_parent == rule_parent:
                exp_scaled[pos] *= tid_prior
            elif rule_tactic and "_tactics" in meta_df.columns:
                cand_tacs = str(row.get("_tactics", "")).lower()
                if rule_tactic.lower() in cand_tacs:
                    exp_scaled[pos] *= tactic_prior

    # ── BM25 lexical score ─────────────────────────────────────────────────
    if bm25_weight > 0 and bm25_rerank_width > 0:
        bm25_obj = _get_bm25(meta_df, _FAISS_SCHEMA_VERSION)
        query_tokens = _tokenize(description)
        if query_tokens:
            bm25_scores = bm25_obj.get_scores(query_tokens)  # shape: (N,)
            # normalize
            if bm25_scores.max() > 0:
                bm25_norm = bm25_scores / bm25_scores.max()
            else:
                bm25_norm = bm25_scores
            exp_max = exp_scaled.max()
            if exp_max > 0:
                dense_norm = exp_scaled / exp_max
            else:
                dense_norm = exp_scaled
            bm25_in_faiss_order = bm25_norm[all_indices[0]]
            fused = (1 - bm25_weight) * dense_norm + bm25_weight * bm25_in_faiss_order
            exp_scaled = fused * exp_max  # keep scale

    order = np.argsort(-exp_scaled)
    all_indices_reranked = all_indices[0][order]
    all_sims_reranked    = all_sims[order]
    exp_scaled_reranked  = exp_scaled[order]

    if cross_encoder is not None and ce_weight > 0 and ce_rerank_width > 0:
        wide = min(ce_rerank_width, len(all_indices_reranked))
        wide_idx = all_indices_reranked[:wide]
        pairs = []
        for orig_idx in wide_idx:
            row = meta_df.iloc[orig_idx]
            tname = str(row.get("_tname", "")).strip()
            tdesc = str(row.get("_desc_clean", "")).strip()[:1500]
            cand_text = f"{tname}: {tdesc}" if tname else tdesc
            pairs.append([description[:1500], cand_text])
        try:
            ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"    [ce-err] {e}; skipping CE rerank")
            ce_scores = None

        if ce_scores is not None:
            ce_scores = np.array(ce_scores, dtype=np.float32)
            ce_norm = 1.0 / (1.0 + np.exp(-ce_scores))

            bi_norm = exp_scaled_reranked[:wide] / exp_scaled_reranked[:wide].max()
            # log(combined) = (1-w)·log(bi) + w·log(ce)
            eps = 1e-9
            log_combined = ((1 - ce_weight) * np.log(bi_norm + eps)
                            + ce_weight * np.log(ce_norm + eps))
            combined = np.exp(log_combined)

            sub_order = np.argsort(-combined)
            new_wide_idx = wide_idx[sub_order]
            new_wide_sims = all_sims_reranked[:wide][sub_order]
            new_wide_exp  = combined[sub_order] * exp_scaled_reranked[:wide].max()

            all_indices_reranked = np.concatenate([new_wide_idx, all_indices_reranked[wide:]])
            all_sims_reranked    = np.concatenate([new_wide_sims, all_sims_reranked[wide:]])
            exp_scaled_reranked  = np.concatenate([new_wide_exp, exp_scaled_reranked[wide:]])

    if signature_weight > 0 and signature_rerank_width > 0:
        sig_map = _load_tid_signatures()
        if sig_map:
            wide = min(signature_rerank_width, len(all_indices_reranked))
            wide_idx = all_indices_reranked[:wide]
            sig_boosts = np.ones(wide, dtype=np.float32)
            for pos, orig_idx in enumerate(wide_idx):
                tid = str(meta_df.iloc[orig_idx].get("_tid", ""))
                s = _signature_match_score(description, tid, sig_map)
                sig_boosts[pos] = 1.0 + signature_weight * s
            boosted = exp_scaled_reranked[:wide] * sig_boosts
            sub_order = np.argsort(-boosted)
            new_wide_idx  = wide_idx[sub_order]
            new_wide_sims = all_sims_reranked[:wide][sub_order]
            new_wide_exp  = boosted[sub_order]
            all_indices_reranked = np.concatenate([new_wide_idx, all_indices_reranked[wide:]])
            all_sims_reranked    = np.concatenate([new_wide_sims, all_sims_reranked[wide:]])
            exp_scaled_reranked  = np.concatenate([new_wide_exp, exp_scaled_reranked[wide:]])

    # ── (A1) Family-consensus boost ──────────────────────────────────────────
    if family_boost > 0 and family_boost_width > 0:
        wide = min(family_boost_width, len(all_indices_reranked))
        wide_idx = all_indices_reranked[:wide]
        parents = []
        for orig_idx in wide_idx:
            tid = str(meta_df.iloc[orig_idx].get("_tid", ""))
            parents.append(tid.split(".", 1)[0] if "." in tid else tid)
        from collections import Counter as _C
        pcount = _C(parents)
        fam_boosts = np.ones(wide, dtype=np.float32)
        for pos, p in enumerate(parents):
            shared = pcount[p] - 1
            fam_boosts[pos] = 1.0 + family_boost * shared
        boosted = exp_scaled_reranked[:wide] * fam_boosts
        sub_order = np.argsort(-boosted)
        new_wide_idx  = wide_idx[sub_order]
        new_wide_sims = all_sims_reranked[:wide][sub_order]
        new_wide_exp  = boosted[sub_order]
        all_indices_reranked = np.concatenate([new_wide_idx, all_indices_reranked[wide:]])
        all_sims_reranked    = np.concatenate([new_wide_sims, all_sims_reranked[wide:]])
        exp_scaled_reranked  = np.concatenate([new_wide_exp, exp_scaled_reranked[wide:]])

    Z = exp_scaled_reranked.sum()
    all_probs = exp_scaled_reranked / Z

    top_k_indices = all_indices_reranked[:k]
    top_k_probs   = all_probs[:k]
    top_k_sims    = all_sims_reranked[:k]

    def _first_tactic(raw: str) -> str:
        raw = (raw or "").strip()
        if not raw or raw.lower() == "nan":
            return ""
        return raw.split(",")[0].strip()

    candidates = [
        {
            "rank":           rank,
            "technique_id":   str(meta_df.iloc[idx].get("_tid",        "N/A")),
            "technique_name": str(meta_df.iloc[idx].get("_tname",      "N/A")),
            "tactic":         _first_tactic(str(meta_df.iloc[idx].get("_tactics", ""))),
            "similarity":     round(float(sim), 4),
            "p_ttp":          round(float(prob), 6),
            "description":    str(meta_df.iloc[idx].get("_desc_clean", ""))[:300],
        }
        for rank, (sim, prob, idx)
        in enumerate(zip(top_k_sims, top_k_probs, top_k_indices), start=1)
    ]

    margin = float(top_k_probs[0] - top_k_probs[1]) if len(top_k_probs) >= 2 else 1.0

    return candidates, round(margin, 6)


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def analyze(
    all_features: list,
    mitre_csv_path: str,
    gemini_api_key: str,
    cache_dir: Optional[str | Path] = None,
    use_llm: bool = True,
    cross_encoder=None,
    ce_rerank_width: int = 20,
    ce_weight: float = 0.6,
    bm25_weight: float = 0.3,
    bm25_rerank_width: int = 30,
    tid_prior: float = 1.15,
    tactic_prior: float = 1.05,
    signature_weight: float = 0.0,
    signature_rerank_width: int = 10,
    family_boost: float = 0.0,
    family_boost_width: int = 10,
) -> list[dict]:
    """feature ...description + Top-K ...technique ...

    """
    embed_model = get_embed_model()
    index, meta_df = build_or_load_faiss_index(mitre_csv_path, embed_model, cache_dir)

    # Pre-build BM25 index once (shared across all groups)
    _bm25_weight = bm25_weight
    _bm25_width = bm25_rerank_width
    _tid_prior = tid_prior
    _tactic_prior = tactic_prior
    if _bm25_weight > 0:
        _get_bm25(meta_df, _FAISS_SCHEMA_VERSION)

    print(f"  ...: {len(all_features)}..."
          f"(CE={'on' if cross_encoder else 'off'}, "
          f"BM25 weight={_bm25_weight}, rule_prior={_tid_prior})")
    results = []
    n_cached = 0

    if not use_llm:
        from experiments.ablation.helpers import feature_to_text

    for feat in all_features:
        gid = feat["group_id"]
        tid = feat["technique_id"]
        print(f"\n  → [{gid}] {tid}")

        if use_llm:
            desc, was_cached = generate_description_cached(feat, gemini_api_key, cache_dir)
            if was_cached:
                n_cached += 1
        else:
            desc = feature_to_text(feat)
            was_cached = False
        similar, conf_margin = search_similar(
            desc, index, meta_df, embed_model,
            rule_tid=tid,
            cross_encoder=cross_encoder,
            ce_rerank_width=ce_rerank_width,
            ce_weight=ce_weight,
            bm25_weight=_bm25_weight,
            bm25_rerank_width=_bm25_width,
            tid_prior=_tid_prior,
            tactic_prior=_tactic_prior,
            signature_weight=signature_weight,
            signature_rerank_width=signature_rerank_width,
            family_boost=family_boost,
            family_boost_width=family_boost_width,
        )

        results.append({
            "group_id":              gid,
            "technique_id":          tid,
            "technique_name":        feat["technique_name"],
            "confidence":            feat["confidence"],
            "confidence_margin":     conf_margin,
            "generated_description": desc,
            "similar_techniques":    similar,
            "all_idxs":              feat.get("all_idxs", []),
        })

        tag = " [cached]" if was_cached else ""
        print(f"  [...]{tag}\n  {desc}")
        print(f"\n  [...Top-{TOP_K}]  (confidence margin={conf_margin:.4f})")
        for s in similar:
            match = " ← MATCH" if s["technique_id"] == tid else ""
            print(f"  {s['rank']}. {s['technique_id']} {s['technique_name']}  "
                  f"sim={s['similarity']:.4f}  p={s['p_ttp']:.4f}{match}")

    print(f"\n  LLM ...: {n_cached}/{len(all_features)}")
    return results
