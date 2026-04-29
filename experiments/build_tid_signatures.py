"""
TID → artifact signature 키워드 사전 빌드.

Technique Rule/*.json 에는 anchor 룰 작성 과정에서 이미 TID 별 결정적 키워드
(CommandLine 의 `contains_any`, TargetImage 의 `contains`, 등) 리스트가 정리되어
있다. 이들을 TID → list[str] 형태로 묶어 JSON 으로 저장, downstream rerank 단계에서
behavior description 과 lexical overlap 을 계산하는 데 사용한다.

사용:
  python experiments/build_tid_signatures.py
출력:
  TTP_Data/tid_signatures.json
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULE_DIR = ROOT / "Technique Rule"
OUT_PATH = ROOT / "TTP_Data" / "tid_signatures.json"


def _collect_strings(obj, acc: list[str]) -> None:
    """dict/list 를 재귀적으로 훑어 `contains`, `contains_any`, `equals` 의
    리스트/문자열 값만 모은다. YAML-style 룰 포맷에서 키워드 시그니처를 추출."""
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
    """중복 제거 + 소문자 + 공백 정리. 영문/숫자 3자 이하나 설명조 제거."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        t2 = t.strip().lower()
        if not t2 or len(t2) < 3:
            continue
        # 너무 일반적인 단어 제거
        if t2 in {"exe", "dll", "cmd", "the", "and", "any", "sys"}:
            continue
        # "note" 같은 메타 필드 텍스트는 공백 포함이 많음 — 공백 없는 것 우선, 있으면 짧은 것만.
        if " " in t2 and len(t2) > 60:
            continue
        if t2 not in seen:
            seen.add(t2)
            out.append(t2)
    return out


def _file_to_tid(stem: str) -> str:
    """파일명에서 TID 추출. 'T1003_001_v2_std' → 'T1003.001'. 'T1082' → 'T1082'."""
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
        # 같은 TID 로 매핑되는 파일이 여럿 있으면 union
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
