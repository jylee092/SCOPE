"""


  python experiments/build_tid_signatures.py
  TTP_Data/tid_signatures.json
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULE_DIR = ROOT / "Technique Rule"
OUT_PATH = ROOT / "TTP_Data" / "tid_signatures.json"


def _collect_strings(obj, acc: list[str]) -> None:
    """Recursively walk a dict/list rule structure and collect keyword
    signatures from string-valued fields under `contains`, `contains_any`,
    `equals`, `endswith`, `startswith`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("contains", "contains_any", "equals", "endswith", "startswith"):
                if isinstance(v, list):
                    for s in v:
                        if isinstance(s, str):
                            acc.append(s)
                elif isinstance(v, str):
                    acc.append(v)
            else:
                _collect_strings(v, acc)
    elif isinstance(obj, list):
        for x in obj:
            _collect_strings(x, acc)


def _normalize(tokens: list[str]) -> list[str]:
    """...+ ...+ ...3..."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        t2 = t.strip().lower()
        if not t2 or len(t2) < 3:
            continue
        if t2 in {"exe", "dll", "cmd", "the", "and", "any", "sys"}:
            continue
        if " " in t2 and len(t2) > 60:
            continue
        if t2 not in seen:
            seen.add(t2)
            out.append(t2)
    return out


def _file_to_tid(stem: str) -> str:
    """...TID ...'T1003_001_v2_std' → 'T1003.001'. 'T1082' → 'T1082'."""
    m = re.match(r"(T\d+)(?:_(\d{3}))?", stem)
    if not m:
        return stem
    head = m.group(1)
    sub = m.group(2)
    return f"{head}.{sub}" if sub else head


def main():
    signatures: dict[str, list[str]] = {}
    for fp in sorted(RULE_DIR.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [skip] {fp.name}: {e}")
            continue
        tid = _file_to_tid(fp.stem)
        buf: list[str] = []
        _collect_strings(data, buf)
        sigs = _normalize(buf)
        if not sigs:
            print(f"  [empty] {fp.name} ({tid})")
            continue
        if tid in signatures:
            merged = list(dict.fromkeys(signatures[tid] + sigs))
            signatures[tid] = merged
        else:
            signatures[tid] = sigs
        print(f"  {tid:12s}  ← {fp.name:40s}  ({len(sigs)} sigs)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(signatures, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {OUT_PATH} ({len(signatures)} TIDs)")


if __name__ == "__main__":
    main()
