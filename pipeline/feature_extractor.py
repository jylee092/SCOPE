"""

          network, persistence, evasion
"""
from __future__ import annotations

import base64
import re
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def _v(row: pd.Series, *keys) -> Optional[str]:
    """..."""
    for k in keys:
        val = row.get(k)
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        s = str(val).strip()
        if s:
            return s
    return None


def _basename(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return path.replace("/", "\\").split("\\")[-1]


def _contains_any(text: Optional[str], patterns: list[str]) -> list[str]:
    if not text:
        return []
    t = text.lower()
    return [p for p in patterns if p.lower() in t]


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
_PERSISTENCE_REG_KEYWORDS = [
    "currentversion\\run", "currentversion\\runonce",
    "currentcontrolset\\services", "userinitmprlogonscript",
    "ms-settings\\shell\\open\\command", "hklm\\security",
    "control\\lsa", "winlogon", "image file execution options",
    "appinit_dlls", "active setup", "policies\\explorer\\run",
]

_NOISE_DROP_PATTERNS = [
    "__psscriptpolicytest_", "moduleanalysiscache",
]

_LOG_CLEAR_EIDS = {1102, 104}

_OBFUSCATION_CMDLINE_PATTERNS = [
    "-enc ", "-encodedcommand", "-ec ",
    "frombase64string", "tobase64string",
    "invoke-expression", " iex ",
    "[convert]::", "[reflection.assembly]",
    "amsiutils", "amsiinitialized",
]

_NET_DST_IP_KEYS   = ("DestinationIp", "DestAddress", "DestIp", "dst_ip")
_NET_SRC_IP_KEYS   = ("SourceIp", "SourceAddress", "src_ip")
_NET_DST_PORT_KEYS = ("DestinationPort", "DestPort", "dst_port")
_NET_SRC_PORT_KEYS = ("SourcePort", "src_port")

_PORT_SERVICE_HINT = {
    80: "HTTP", 443: "HTTPS", 8080: "HTTP-Alt (Proxy/C2)",
    8443: "HTTPS-Alt (C2)", 4443: "HTTPS-Alt (C2)",
    9001: "Tor/C2", 21: "FTP", 22: "SSH", 23: "Telnet",
    53: "DNS", 3389: "RDP", 445: "SMB",
}

_NET_EIDS    = {3, 5154, 5155, 5156, 5157, 5158, 22}
_LISTEN_EIDS = {5154, 5158}
_DNS_EIDS    = {22}


# ──────────────────────────────────────────────────────────────────────────────
# ① Execution Context
# ──────────────────────────────────────────────────────────────────────────────
def _extract_execution_context(grp: pd.DataFrame) -> dict:
    chains: list[dict] = []
    seen: set[tuple] = set()

    for _, r in grp[grp["EventID"] == 1].iterrows():
        child  = _basename(_v(r, "Image"))
        parent = _basename(_v(r, "ParentImage"))
        key = (parent, child, "spawn")
        if child and key not in seen:
            chains.append({
                "relation": "spawn",
                "parent_image": parent, "child_image": child,
                "integrity_level": _v(r, "IntegrityLevel"),
                "parent_guid": _v(r, "ParentProcessGuid"),
                "child_guid":  _v(r, "ProcessGuid"),
                "cmdline": _v(r, "CommandLine"),
                "source_eid": 1,
            })
            seen.add(key)

    for _, r in grp[grp["EventID"] == 4688].iterrows():
        child  = _basename(_v(r, "NewProcessName"))
        parent = _basename(_v(r, "ParentProcessName"))
        key = (parent, child, "spawn")
        if child and key not in seen:
            chains.append({
                "relation": "spawn",
                "parent_image": parent, "child_image": child,
                "integrity_level": None, "parent_guid": None, "child_guid": None,
                "cmdline": _v(r, "CommandLine"),
                "source_eid": 4688,
            })
            seen.add(key)

    for _, r in grp[grp["EventID"] == 10].iterrows():
        src = _basename(_v(r, "SourceImage"))
        tgt = _basename(_v(r, "TargetImage"))
        key = (src, tgt, "access")
        if src and tgt and src != tgt and key not in seen:
            chains.append({
                "relation": "access",
                "parent_image": src, "child_image": tgt,
                "granted_access": _v(r, "GrantedAccess"),
                "source_guid": _v(r, "SourceProcessGUID"),
                "target_guid": _v(r, "TargetProcessGUID"),
                "source_eid": 10,
            })
            seen.add(key)

    if not chains:
        seen_imgs: set[str] = set()
        for _, r in grp.iterrows():
            img = _basename(_v(r, "Image", "Application"))
            if img and img not in seen_imgs:
                chains.append({
                    "relation": "observed",
                    "child_image": img,
                    "parent_image": _basename(_v(r, "ParentImage")),
                    "source_eid": int(r["EventID"]),
                })
                seen_imgs.add(img)

    return {"process_chains": chains, "chain_depth": len(chains)}


# ──────────────────────────────────────────────────────────────────────────────
# ② Command / Script
# ──────────────────────────────────────────────────────────────────────────────
def _has_obfuscation(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(p in t for p in _OBFUSCATION_CMDLINE_PATTERNS):
        return True
    m = re.search(r"[A-Za-z0-9+/]{40,}={0,2}", text)
    if m:
        try:
            base64.b64decode(m.group() + "==")
            return True
        except Exception:
            pass
    if re.search(r"\^{3,}", text):
        return True
    return False


def _extract_command_script(grp: pd.DataFrame) -> dict:
    entries: list[dict] = []
    seen_cls: set[str] = set()

    cl_mask = grp["CommandLine"].notna() if "CommandLine" in grp.columns else pd.Series(False, index=grp.index)
    for _, r in grp[cl_mask].iterrows():
        cl = _v(r, "CommandLine")
        if not cl or cl in seen_cls:
            continue
        seen_cls.add(cl)
        entries.append({
            "source_eid": int(r["EventID"]),
            "image": _basename(_v(r, "Image", "NewProcessName")),
            "cmdline": cl,
            "cmdline_length": len(cl),
            "has_obfuscation": _has_obfuscation(cl),
            "special_chars": [c for c in ("^", "&", "|", "%", "`") if c in cl],
        })

    if "ScriptBlockText" in grp.columns:
        for _, r in grp[grp["EventID"] == 4104].iterrows():
            sb = _v(r, "ScriptBlockText")
            if not sb:
                continue
            entries.append({
                "source_eid": 4104,
                "image": "powershell.exe",
                "cmdline": sb[:2000],
                "cmdline_length": len(sb),
                "has_obfuscation": _has_obfuscation(sb),
                "special_chars": [],
            })

    return {
        "entries": entries,
        "has_obfuscation": any(e["has_obfuscation"] for e in entries),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ③ Identity
# ──────────────────────────────────────────────────────────────────────────────
def _extract_identity(grp: pd.DataFrame) -> dict:
    priority_eids = [1, 4688, 4624, 4672, 4656, 4663]
    for eid in priority_eids:
        rows = grp[grp["EventID"] == eid]
        if rows.empty:
            continue
        r = rows.iloc[0]
        return {
            "source_eid": eid,
            "user":            _v(r, "User", "SubjectUserName", "TargetUserName"),
            "domain":          _v(r, "Domain", "SubjectDomainName", "TargetDomainName"),
            "logon_id":        _v(r, "LogonId", "SubjectLogonId", "TargetLogonId"),
            "integrity_level": _v(r, "IntegrityLevel"),
            "user_sid":        _v(r, "UserSid", "SubjectUserSid", "TargetUserSid"),
            "privilege_list":  _v(r, "PrivilegeList"),
        }

    for _, r in grp.iterrows():
        user = _v(r, "User", "SubjectUserName", "TargetUserName")
        if user:
            return {
                "source_eid": int(r["EventID"]),
                "user": user,
                "domain":          _v(r, "Domain", "SubjectDomainName"),
                "logon_id":        _v(r, "LogonId", "SubjectLogonId"),
                "integrity_level": _v(r, "IntegrityLevel"),
                "user_sid":        _v(r, "UserSid", "SubjectUserSid"),
                "privilege_list":  _v(r, "PrivilegeList"),
            }

    return {
        "source_eid": None, "user": None, "domain": None,
        "logon_id": None, "integrity_level": None,
        "user_sid": None, "privilege_list": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ④ Temporal
# ──────────────────────────────────────────────────────────────────────────────
def _extract_temporal(grp: pd.DataFrame, anchor_idx: int) -> dict:
    anchor_t = (
        grp.loc[anchor_idx, "TimeCreated"]
        if anchor_idx in grp.index else grp["TimeCreated"].min()
    )
    times  = grp["TimeCreated"].sort_values()
    deltas = [(t - anchor_t).total_seconds() for t in times]
    span   = deltas[-1] - deltas[0] if len(deltas) > 1 else 0.0

    gaps = [deltas[i+1] - deltas[i] for i in range(len(deltas)-1)]
    gap_stats = {
        "min_sec":  round(min(gaps), 3) if gaps else None,
        "max_sec":  round(max(gaps), 3) if gaps else None,
        "mean_sec": round(sum(gaps)/len(gaps), 3) if gaps else None,
    }

    eid_counts = {int(k): int(v) for k, v in grp["EventID"].value_counts().to_dict().items()}

    burst_eids: list[dict] = []
    for eid, cnt in eid_counts.items():
        eid_times = sorted(grp[grp["EventID"] == eid]["TimeCreated"].tolist())
        for i in range(len(eid_times) - 2):
            window = (eid_times[i+2] - eid_times[i]).total_seconds()
            if window <= 10:
                burst_eids.append({"eid": eid, "count": cnt, "window_sec": round(window, 2)})
                break

    return {
        "anchor_time":             anchor_t.isoformat(),
        "window_start_delta_sec":  round(deltas[0], 2),
        "window_end_delta_sec":    round(deltas[-1], 2),
        "span_sec":                round(span, 2),
        "total_events":            len(deltas),
        "event_density_per_10sec": round(len(deltas) / span * 10, 2) if span >= 1.0 else None,
        "eid_counts":              eid_counts,
        "inter_event_gap":         gap_stats,
        "burst_detected":          burst_eids,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ⑤ Network
# ──────────────────────────────────────────────────────────────────────────────
def _extract_network(grp: pd.DataFrame) -> dict:
    connections: list[dict] = []
    net_rows = grp[grp["EventID"].isin(_NET_EIDS)] if not grp.empty else pd.DataFrame()

    for _, r in net_rows.iterrows():
        eid = int(r["EventID"])

        dst_ip   = _v(r, *_NET_DST_IP_KEYS)
        src_ip   = _v(r, *_NET_SRC_IP_KEYS)
        dst_port_str = _v(r, *_NET_DST_PORT_KEYS)
        src_port_str = _v(r, *_NET_SRC_PORT_KEYS)

        try:    dst_port = int(float(dst_port_str)) if dst_port_str else None
        except (ValueError, TypeError): dst_port = None
        try:    src_port = int(float(src_port_str)) if src_port_str else None
        except (ValueError, TypeError): src_port = None

        proto_raw = _v(r, "Protocol")
        protocol  = "TCP" if proto_raw == "6" else ("UDP" if proto_raw == "17" else proto_raw)
        img = _basename(_v(r, "Image", "Application"))

        entry: dict = {
            "source_eid": eid, "image": img,
            "src_ip": src_ip, "dst_ip": dst_ip,
            "src_port": src_port, "dst_port": dst_port,
            "protocol": protocol, "direction": None,
            "service_hint": _PORT_SERVICE_HINT.get(dst_port or src_port),
        }

        if eid in _LISTEN_EIDS:
            entry["direction"]   = "listen"
            entry["listen_port"] = src_port
        elif eid in _DNS_EIDS:
            entry["direction"]  = "dns_query"
            entry["query_name"] = _v(r, "QueryName", "QueryResults")
        else:
            entry["direction"] = "outbound"

        connections.append(entry)

    return {"connections": connections}


# ──────────────────────────────────────────────────────────────────────────────
# ⑥ Persistence
# ──────────────────────────────────────────────────────────────────────────────
def _extract_persistence(grp: pd.DataFrame) -> dict:
    reg_signals: list[dict] = []
    reg_noise:   list[dict] = []

    for _, r in grp[grp["EventID"].isin([13, 14])].iterrows():
        obj = _v(r, "TargetObject")
        if not obj:
            continue
        obj_lower = obj.lower()
        entry = {
            "source_eid":   int(r["EventID"]),
            "target_object": obj,
            "details":      _v(r, "Details"),
            "process_guid": _v(r, "ProcessGuid"),
            "image":        _basename(_v(r, "Image")),
        }
        if any(kw in obj_lower for kw in _PERSISTENCE_REG_KEYWORDS):
            reg_signals.append(entry)
        else:
            reg_noise.append(entry)

    dropped_files: list[dict] = []
    file_rows = grp[grp["EventID"] == 11] if not grp.empty else pd.DataFrame()
    for _, r in file_rows.iterrows():
        path = _v(r, "TargetFilename")
        if not path:
            continue
        if any(p in path.lower() for p in _NOISE_DROP_PATTERNS):
            continue
        dropped_files.append({
            "source_eid": 11, "path": path,
            "image": _basename(_v(r, "Image")),
        })

    return {
        "registry_signals":     reg_signals,
        "registry_noise":       reg_noise,
        "registry_noise_count": len(reg_noise),
        "dropped_files":        dropped_files,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ⑦ Evasion
# ──────────────────────────────────────────────────────────────────────────────
def _extract_evasion(grp: pd.DataFrame) -> dict:
    log_cleared: list[dict] = []
    for _, r in grp[grp["EventID"].isin(_LOG_CLEAR_EIDS)].iterrows():
        log_cleared.append({
            "source_eid": int(r["EventID"]),
            "message":    _v(r, "Message"),
            "user":       _v(r, "SubjectUserName", "User"),
        })

    deleted_files: list[dict] = []
    if not grp.empty and 23 in grp["EventID"].values:
        for _, r in grp[grp["EventID"] == 23].iterrows():
            path = _v(r, "TargetFilename")
            if not path:
                continue
            if any(p in path.lower() for p in _NOISE_DROP_PATTERNS):
                continue
            deleted_files.append({
                "source_eid": 23, "path": path,
                "image": _basename(_v(r, "Image")),
            })

    obf_entries: list[dict] = []
    if "CommandLine" in grp.columns:
        for _, r in grp[grp["CommandLine"].notna()].iterrows():
            cl = _v(r, "CommandLine")
            if cl and _has_obfuscation(cl):
                obf_entries.append({
                    "source_eid": int(r["EventID"]),
                    "image":   _basename(_v(r, "Image", "NewProcessName")),
                    "cmdline": cl[:500],
                })

    return {
        "log_cleared":         log_cleared,
        "deleted_files":       deleted_files,
        "obfuscated_cmdlines": obf_entries,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def _extract_anchor_detail(df: pd.DataFrame, anchor_idx) -> dict:
    """Anchor event...LLM...rule ..."""
    if anchor_idx is None or anchor_idx not in df.index:
        return {}
    r = df.loc[anchor_idx]
    return {
        "eid":           int(r.get("EventID", 0)) if pd.notna(r.get("EventID", None)) else None,
        "time":          str(r.get("TimeCreated", "")),
        "image":         _basename(_v(r, "Image", "NewProcessName", "Application")),
        "parent_image":  _basename(_v(r, "ParentImage", "ParentProcessName")),
        "cmdline":       (_v(r, "CommandLine") or "")[:400],
        "target_object": (_v(r, "TargetObject") or "")[:200],
        "target_image":  _basename(_v(r, "TargetImage")),
        "target_file":   (_v(r, "TargetFilename") or "")[:200],
        "granted_access": _v(r, "GrantedAccess"),
        "query_name":    _v(r, "QueryName"),
        "dst_ip":        _v(r, *_NET_DST_IP_KEYS),
        "dst_port":      _v(r, *_NET_DST_PORT_KEYS),
        "user":          _v(r, "User", "SubjectUserName", "TargetUserName"),
        "integrity":     _v(r, "IntegrityLevel"),
    }


def extract_features(group: dict, df: pd.DataFrame) -> dict:
    """...7...Feature ..."""
    idxs       = group.get("all_idxs") or group.get("confirmed_idxs", [])
    anchor_idx = group.get("anchor_idx")

    valid_idxs = [i for i in idxs if i in df.index]
    grp = df.loc[valid_idxs].copy() if valid_idxs else pd.DataFrame(columns=df.columns)

    return {
        "group_id":       group.get("group_id", ""),
        "technique_id":   group.get("technique_id", ""),
        "technique_name": group.get("technique_name", ""),
        "anchor_eid":     group.get("anchor_eid"),
        "anchor_detail":  _extract_anchor_detail(df, anchor_idx),
        "confidence":     group.get("confidence", 0.0),
        "all_idxs":       group.get("all_idxs", []),
        "features": {
            "execution_context": _extract_execution_context(grp),
            "command_script":    _extract_command_script(grp),
            "identity":          _extract_identity(grp),
            "temporal":          _extract_temporal(grp, anchor_idx),
            "network":           _extract_network(grp),
            "persistence":       _extract_persistence(grp),
            "evasion":           _extract_evasion(grp),
        },
    }


def extract_all(groups: list[dict], df: pd.DataFrame) -> list[dict]:
    results = []
    for g in groups:
        feat = extract_features(g, df)
        results.append(feat)
        print(f"  [{feat['group_id']}] {feat['technique_id']}  confidence={feat['confidence']}")
    print(f"\n...{len(results)}...")
    return results
