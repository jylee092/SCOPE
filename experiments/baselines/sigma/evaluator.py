"""
Sigma rule evaluator -- in-memory matching of SigmaHQ YAML rules against
normalized Mordor/winlogbeat events.

Scope: only Windows rules (rules/windows/**), supporting the modifiers and
condition forms actually used in that subset.

Modifiers supported: contains, startswith, endswith, re, all, cased, i,
                     plus equality (no modifier).
Condition forms supported: NAME, 'not' EXPR, 'and'/'or', '(...)',
                           '1 of GLOB', 'all of GLOB', '<n> of GLOB' (n>=1 → at-least-1),
                           'them' / '1 of them' / 'all of them'.

The evaluator is intentionally Python-only; it relies on the field
normalization already performed by pipeline.data_loader._normalize_winlogbeat.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


# ----------------------------------------------------------------------------
# Logsource → EventID / channel filtering
# ----------------------------------------------------------------------------

# Sigma category → likely Sysmon EIDs (Mordor logs use Sysmon almost everywhere).
# Values are *acceptable* EIDs; an event matches the category iff its EventID is
# in the set, OR the set is empty (catch-all).
_CATEGORY_EIDS: dict[str, set[int]] = {
    "process_creation":      {1, 4688},
    "file_event":            {11},
    "file_change":           {2},
    "file_delete":           {23, 26, 4660},
    "file_access":           {4663},
    "file_rename":           {29},
    "registry_event":        {12, 13, 14},
    "registry_set":          {13},
    "registry_add":          {12},
    "registry_delete":       {12},
    "registry_rename":       {14},
    "network_connection":    {3, 5156},
    "image_load":            {7},
    "dns_query":             {22, 5158},
    "pipe_created":          {17, 18},
    "process_access":        {10},
    "create_remote_thread":  {8},
    "create_stream_hash":    {15},
    "driver_load":           {6},
    "ps_module":             {4103},
    "ps_script":             {4104},
    "ps_classic_start":      {400, 600},
    "raw_access_thread":     {9},
    "process_tampering":     {25},
    "wmi_event":             {19, 20, 21},
    "sysmon_status":         {4, 16},
}

# Sigma service → channel name fragment (case-insensitive substring match
# against either Channel or SourceName).
_SERVICE_CHANNEL: dict[str, tuple[str, ...]] = {
    "security":          ("Security",),
    "system":            ("System",),
    "application":       ("Application",),
    "sysmon":            ("Sysmon",),
    "powershell":        ("PowerShell",),
    "powershell-classic": ("PowerShell",),
    "taskscheduler":     ("TaskScheduler",),
    "wmi":               ("WMI-Activity",),
    "ntlm":              ("NTLM",),
    "dns-server":        ("DNS-Server",),
    "dns-server-analytic": ("DNS-Server",),
    "applocker":         ("AppLocker",),
    "bits-client":       ("Bits-Client",),
    "smbclient-security": ("SmbClient/Security",),
    "smbserver":         ("SmbServer",),
    "lsa-server":        ("LSA",),
    "msexchange-management": ("Exchange",),
    "ldap_debug":        ("LDAP",),
    "ldap-client":       ("LDAP",),
    "ntfs":              ("Ntfs",),
    "kernel-pnp":        ("Kernel-PnP",),
    "kernel-shimengine": ("Kernel-ShimEngine",),
    "openssh":           ("OpenSSH",),
    "windefend":         ("Windows Defender",),
    "firewall-as":       ("Windows Firewall",),
    "appxpackaging-om":  ("AppxPackaging",),
    "appxdeployment-server": ("AppXDeployment-Server",),
    "appmodel-runtime":  ("AppModel-Runtime",),
    "code-integrity":    ("CodeIntegrity",),
    "diagnosis-scripted": ("Diagnosis",),
    "shell-core":        ("Shell-Core",),
    "printservice-admin": ("PrintService",),
    "printservice-operational": ("PrintService",),
    "terminalservices-localsessionmanager": ("TerminalServices",),
    "terminalservices-remoteconnectionmanager": ("TerminalServices",),
    "wmi-activity":      ("WMI-Activity",),
}


# ----------------------------------------------------------------------------
# Rule data class
# ----------------------------------------------------------------------------

@dataclass
class SigmaRule:
    path: Path
    title: str
    rule_id: str
    level: str          # critical/high/medium/low/informational
    status: str         # stable/test/experimental/deprecated/unsupported
    tags: list[str]
    tids: list[str]     # extracted ATT&CK technique IDs (T1003, T1003.001, …)
    tactics: list[str]  # extracted ATT&CK tactic slugs (credential-access, …)
    logsource: dict
    detection: dict
    condition: str
    selection_count: int


_LEVEL_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1,
               "informational": 0, "info": 0}
_STATUS_RANK = {"stable": 3, "test": 2, "experimental": 1,
                "deprecated": 0, "unsupported": 0}


def rule_priority_key(r: SigmaRule) -> tuple[int, int, int, str]:
    """Sort key (descending). Higher tuple = better."""
    return (
        _LEVEL_RANK.get((r.level or "medium").lower(), 2),
        _STATUS_RANK.get((r.status or "experimental").lower(), 1),
        r.selection_count,
        r.rule_id,  # alpha tiebreak for stable order
    )


# ----------------------------------------------------------------------------
# Rule loading
# ----------------------------------------------------------------------------

_TAG_TID_RE = re.compile(r"^t\d{4}(?:\.\d{3})?$")


def _extract_attack_tags(tags: Iterable[str]) -> tuple[list[str], list[str]]:
    tids, tactics = [], []
    for t in tags or []:
        if not isinstance(t, str):
            continue
        s = t.strip().lower()
        if not s.startswith("attack."):
            continue
        body = s[len("attack."):]
        if _TAG_TID_RE.match(body):
            tids.append(body.upper())
        elif body and not body.startswith(("s0", "g0", "car.")):
            tactics.append(body)
    return tids, tactics


def load_rules(rules_dir: Path,
               require_attack_tag: bool = True) -> list[SigmaRule]:
    """Load all *.yml under rules_dir; skip rules that fail to parse or
    do not carry any attack.tNNNN tag (controlled by `require_attack_tag`)."""
    out: list[SigmaRule] = []
    skipped = 0
    for p in sorted(rules_dir.rglob("*.yml")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            skipped += 1
            continue
        if not isinstance(data, dict):
            skipped += 1
            continue
        if (data.get("status") or "").lower() == "deprecated":
            skipped += 1
            continue
        detection = data.get("detection") or {}
        if not isinstance(detection, dict) or not detection:
            skipped += 1
            continue
        condition = detection.get("condition") or ""
        if isinstance(condition, list):
            condition = " or ".join(str(c) for c in condition if c)
        condition = str(condition).strip()
        if not condition:
            skipped += 1
            continue
        tids, tactics = _extract_attack_tags(data.get("tags") or [])
        if require_attack_tag and not tids:
            skipped += 1
            continue
        sel_count = sum(1 for k in detection.keys() if k != "condition")
        out.append(SigmaRule(
            path=p,
            title=str(data.get("title") or ""),
            rule_id=str(data.get("id") or p.stem),
            level=str(data.get("level") or "medium"),
            status=str(data.get("status") or "experimental"),
            tags=list(data.get("tags") or []),
            tids=tids,
            tactics=tactics,
            logsource=dict(data.get("logsource") or {}),
            detection=detection,
            condition=condition,
            selection_count=sel_count,
        ))
    return out


# ----------------------------------------------------------------------------
# Logsource matching
# ----------------------------------------------------------------------------

def _logsource_matches(logsource: dict, event: dict) -> bool:
    """Quick coarse filter -- discard events that the rule's logsource cannot
    possibly produce. We are deliberately permissive on missing fields."""
    if not logsource:
        return True
    product = (logsource.get("product") or "").lower()
    if product and product != "windows":
        return False
    cat = (logsource.get("category") or "").lower()
    svc = (logsource.get("service") or "").lower()

    eid = event.get("EventID")
    try:
        eid = int(eid) if eid is not None else None
    except (TypeError, ValueError):
        eid = None

    if cat:
        allowed = _CATEGORY_EIDS.get(cat)
        if allowed is not None and eid not in allowed:
            return False

    if svc:
        wanted = _SERVICE_CHANNEL.get(svc)
        if wanted:
            chan = str(event.get("Channel") or "")
            src  = str(event.get("SourceName") or "")
            target = (chan + " " + src).lower()
            if not any(w.lower() in target for w in wanted):
                return False
    return True


# ----------------------------------------------------------------------------
# Field value matching with modifiers
# ----------------------------------------------------------------------------

def _to_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float):
        try:
            import math
            if math.isnan(v):
                return None
        except Exception:
            pass
    return str(v)


def _atom_match(field_val: Any, mods: list[str], expected: Any) -> bool:
    """Match a single (field, modifier-stack, expected) triple.
    `expected` may be a scalar or a list (list = OR semantics, unless 'all')."""
    sv = _to_str(field_val)
    if sv is None:
        return False
    cased = "cased" in mods
    if not cased:
        sv_cmp = sv.lower()
    else:
        sv_cmp = sv

    use_all = "all" in mods
    items = expected if isinstance(expected, list) else [expected]
    items = [_to_str(x) for x in items]
    items = [x for x in items if x is not None]
    if not items:
        return False
    if not cased:
        items_cmp = [x.lower() for x in items]
    else:
        items_cmp = items

    def _one(item: str) -> bool:
        if "contains" in mods:
            return item in sv_cmp
        if "startswith" in mods:
            return sv_cmp.startswith(item)
        if "endswith" in mods:
            return sv_cmp.endswith(item)
        if "re" in mods:
            flags = 0 if cased else re.IGNORECASE
            try:
                return re.search(item, sv, flags) is not None
            except re.error:
                return False
        # equality
        return sv_cmp == item

    if use_all:
        return all(_one(it) for it in items_cmp)
    return any(_one(it) for it in items_cmp)


def _selection_matches(selection: Any, event: dict) -> bool:
    """A 'selection' is either:
       - a dict {field|mod: value, ...}        → all key/value pairs must hold (AND)
       - a list of such dicts                  → at least one item matches (OR)
    """
    if isinstance(selection, list):
        for item in selection:
            if _selection_matches(item, event):
                return True
        return False
    if not isinstance(selection, dict):
        # uncommon: list of strings → keyword search across all fields
        return False
    for key, val in selection.items():
        parts = str(key).split("|")
        field_name = parts[0]
        mods = [m.lower() for m in parts[1:]]
        # nested OR alternative dict for same field via multiple keys is
        # handled implicitly because each key in dict = AND; OR is encoded by
        # putting alternatives in a list at higher level.
        fv = event.get(field_name)
        if not _atom_match(fv, mods, val):
            return False
    return True


# ----------------------------------------------------------------------------
# Condition expression parser
# ----------------------------------------------------------------------------

class _Tokenizer:
    _PAT = re.compile(r"\s*(\(|\)|\b(?:and|or|not|of|them|all|\d+)\b|[A-Za-z_][\w\-\*]*|\S)\s*")
    def __init__(self, text: str):
        self.tokens = [m.group(1) for m in self._PAT.finditer(text)]
        self.pos = 0
    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok


def _eval_condition(condition: str, sel_results: dict[str, bool]) -> bool:
    """Evaluate a Sigma condition expression.

    Recognized forms:
        atom   ::= NAME | '(' expr ')' | quant
        quant  ::= ('1' | 'all' | <n>) 'of' (NAME | GLOB | 'them')
        notExpr::= 'not' notExpr | atom
        andExpr::= notExpr ('and' notExpr)*
        orExpr ::= andExpr ('or' andExpr)*
    """
    tk = _Tokenizer(condition)

    def _glob_keys(pat: str) -> list[str]:
        if pat == "them":
            return list(sel_results.keys())
        if "*" not in pat:
            return [pat]
        rx = re.compile("^" + re.escape(pat).replace(r"\*", ".*") + "$")
        return [k for k in sel_results.keys() if rx.match(k)]

    def parse_quant(qty: str) -> bool:
        # qty in {'1', 'all', '<digits>'}; eat 'of' and then NAME/GLOB/'them'
        if tk.peek() != "of":
            return False
        tk.consume()
        nxt = tk.peek()
        if nxt is None:
            return False
        target = tk.consume()
        keys = _glob_keys(target)
        bools = [sel_results.get(k, False) for k in keys]
        if not bools:
            return False
        if qty == "all":
            return all(bools)
        # numeric: '1', '2', ... → at-least-1 (Sigma uses '1 of' overwhelmingly;
        # treating other numerics as ≥1 is a deliberate, conservative simplification).
        return any(bools)

    def parse_atom() -> bool:
        tok = tk.peek()
        if tok is None:
            return False
        if tok == "(":
            tk.consume()
            v = parse_or()
            if tk.peek() == ")":
                tk.consume()
            return v
        if tok == "all":
            tk.consume()
            return parse_quant("all")
        if tok == "1" or (tok and tok.isdigit()):
            tk.consume()
            return parse_quant(tok)
        # bare NAME or 'them'
        name = tk.consume()
        if name == "them":
            return any(sel_results.values())
        if "*" in name:
            keys = _glob_keys(name)
            return any(sel_results.get(k, False) for k in keys)
        return sel_results.get(name, False)

    def parse_not() -> bool:
        if tk.peek() == "not":
            tk.consume()
            return not parse_not()
        return parse_atom()

    def parse_and() -> bool:
        v = parse_not()
        while tk.peek() == "and":
            tk.consume()
            v = parse_not() and v
        return v

    def parse_or() -> bool:
        v = parse_and()
        while tk.peek() == "or":
            tk.consume()
            v = parse_and() or v
        return v

    try:
        return parse_or()
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Public entry point: evaluate one rule against one event
# ----------------------------------------------------------------------------

def evaluate_rule(rule: SigmaRule, event: dict) -> bool:
    if not _logsource_matches(rule.logsource, event):
        return False
    sel_results: dict[str, bool] = {}
    for k, v in rule.detection.items():
        if k == "condition":
            continue
        sel_results[k] = _selection_matches(v, event)
    if not any(sel_results.values()):
        # short-circuit: if no positive selection, condition is almost always False
        # (rare 'not X' top-level expressions rejected here are not used in rules/windows)
        return False
    return _eval_condition(rule.condition, sel_results)


def match_event(event: dict, rules: list[SigmaRule]) -> list[SigmaRule]:
    """Return rules that match this event, in priority order (highest first)."""
    hits = [r for r in rules if evaluate_rule(r, event)]
    hits.sort(key=rule_priority_key, reverse=True)
    return hits
