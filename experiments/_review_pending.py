"""
Pending 그룹 검토 도구.

사용:
    python experiments/_review_pending.py <scenario-substring>

각 pending 그룹에 대해 컴팩트한 정보 (anchor, rule, key features, sample
cmdlines, lsass access, registry signals 등) 출력.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"


def _norm(v):
    if v is None: return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan","none","") else s


def _basename(p):
    return p.replace("\\","/").split("/")[-1] if p else ""


def fmt_feature_compact(features: dict) -> str:
    """Compact one-line summary of features."""
    bits = []
    chains = features.get("execution_context",{}).get("process_chains") or []
    if chains:
        ch_summary = []
        for c in chains[:4]:
            p = _basename(c.get("parent_image","") or "?")
            ch = _basename(c.get("child_image","") or "?")
            rel = c.get("relation","?")
            ga = c.get("granted_access")
            extra = f"[{ga}]" if ga else ""
            ch_summary.append(f"{p}--{rel}->{ch}{extra}")
        more = f"+{len(chains)-4}" if len(chains)>4 else ""
        bits.append("PC=" + "; ".join(ch_summary) + more)

    cmds = features.get("command_script",{}).get("entries") or []
    if cmds:
        cmd_summary = []
        for c in cmds[:3]:
            img = _basename(c.get("image","") or "?")
            cmd = (c.get("cmdline","") or "").strip()
            cmd_summary.append(f"{img}: {cmd[:120]}")
        more = f" +{len(cmds)-3}" if len(cmds)>3 else ""
        bits.append("CMD=" + " | ".join(cmd_summary) + more)

    reg_sigs = features.get("persistence",{}).get("registry_signals") or []
    if reg_sigs:
        bits.append(f"REG_SIG={reg_sigs[:3]}")
    reg_noise = features.get("persistence",{}).get("registry_noise") or []
    if reg_noise:
        sample = reg_noise[:2]
        cnt = features.get("persistence",{}).get("registry_noise_count") or len(reg_noise)
        bits.append(f"REG_N({cnt})={sample}")

    nets = features.get("network",{}).get("connections") or []
    if nets:
        bits.append(f"NET=[{len(nets)} conn] {[(c.get('destination_ip'), c.get('destination_port')) for c in nets[:3]]}")

    ev = features.get("evasion",{})
    if ev.get("log_cleared"): bits.append(f"EVASION:log_cleared={ev['log_cleared']}")
    if ev.get("deleted_files"): bits.append(f"EVASION:del_files={ev['deleted_files'][:2]}")
    if ev.get("obfuscated_cmdlines"): bits.append("EVASION:obfuscated")

    iden = features.get("identity",{})
    if iden:
        bits.append(f"USR={iden.get('user','?')}/{iden.get('domain','?')} INTEG={iden.get('integrity_level','?')}")

    temp = features.get("temporal",{})
    if temp:
        bits.append(f"T={temp.get('total_events','?')}ev/{temp.get('span_sec','?')}s "
                    f"EID={list((temp.get('eid_counts') or {}).keys())[:6]}")

    return "  " + "\n  ".join(bits) if bits else "  (no features)"


def review_scenario(substr: str) -> None:
    files = list(OUTPUT_DIR.rglob(f"*{substr}*_annotation.json"))
    if not files:
        print(f"No scenario matching: {substr}")
        return
    for ann in files:
        ftr = ann.with_name(ann.name.replace("_annotation.json","_feature_result.json"))
        if not ftr.exists(): continue
        with open(ann,"r",encoding="utf-8") as f: a = json.load(f)
        with open(ftr,"r",encoding="utf-8") as f: fd = json.load(f)
        fmap = {g["group_id"]: g.get("features",{}) for g in fd}
        scenario = a.get("scenario", ann.parent.name)

        pending = [g for g in a.get("groups",[]) if g.get("gt_is_true_positive") is None]
        if not pending: continue

        print(f"\n{'='*100}")
        print(f"  SCENARIO: {scenario}  (pending {len(pending)} groups)")
        print(f"{'='*100}")
        for i, g in enumerate(pending):
            anchor = g.get("anchor",{}) or {}
            print(f"\n[#{i:>3}] gid={g['group_id']}  rule={g.get('rule_technique_id')}  "
                  f"conf={g.get('confidence')}  num_events={g.get('num_events')}")
            img = _norm(anchor.get('Image'))
            cmd = _norm(anchor.get('CommandLine'))
            par = _norm(anchor.get('ParentImage'))
            tgt = _norm(anchor.get('TargetObject'))
            if img: print(f"  anchor.Image     = {img}")
            if cmd: print(f"  anchor.CommandLine = {cmd[:200]}")
            if par: print(f"  anchor.ParentImage = {par}")
            if tgt: print(f"  anchor.TargetObject = {tgt[:200]}")
            print(fmt_feature_compact(fmap.get(g["group_id"], {})))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python _review_pending.py <scenario-substring>")
        sys.exit(1)
    review_scenario(sys.argv[1])
