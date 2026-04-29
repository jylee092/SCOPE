"""
Section 2. 그룹 생성 (Rule JSON 기반 자동 그룹핑)

공개 API
--------
load_rules(rule_folder)             -> list[dict]
run_grouping(df, rule_list, ...)    -> list[dict]
merge_shared_supporting(groups, df) -> list[dict]
print_groups(groups)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1. 유틸
# ──────────────────────────────────────────────────────────────────────────────
def _str_val(val) -> Optional[str]:
    """NaN/None → None, 그 외 → strip된 소문자 문자열."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s.lower() if s else None


# ──────────────────────────────────────────────────────────────────────────────
# Shared Entity 추출 — file path / registry key / network / pipe
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_reg(path: str) -> str:
    """레지스트리 경로 정규화 (HKLM/HKCU 축약 통일)."""
    lp = path.lower().strip()
    lp = re.sub(r"^\\registry\\machine", "hklm", lp)
    lp = re.sub(r"^\\registry\\user\\[^\\]+", "hkcu", lp)
    lp = re.sub(r"^hkey_local_machine", "hklm", lp)
    lp = re.sub(r"^hkey_current_user", "hkcu", lp)
    return lp


def _collect_artifacts(row: pd.Series) -> dict[str, set[str]]:
    """이벤트 row에서 공유 가능한 artifact 추출.

    카테고리:
      file_written — TargetFilename (drop한 파일)
      image_run    — Image (실행된 바이너리)
      registry     — TargetObject
      network      — DestinationIp / DestinationHostname
      pipe         — PipeName
    """
    arts: dict[str, set[str]] = {
        "file_written": set(),
        "image_run":    set(),
        "registry":     set(),
        "network":      set(),
        "pipe":         set(),
    }

    tf = _str_val(row.get("TargetFilename"))
    if tf:
        arts["file_written"].add(tf)

    img = _str_val(row.get("Image"))
    if img:
        arts["image_run"].add(img)

    img_loaded = _str_val(row.get("ImageLoaded"))
    if img_loaded:
        arts["image_run"].add(img_loaded)

    to = _str_val(row.get("TargetObject"))
    if to:
        arts["registry"].add(_normalize_reg(to))

    for f in ("DestinationIp", "DestinationHostname"):
        v = _str_val(row.get(f))
        if v:
            arts["network"].add(v)

    pn = _str_val(row.get("PipeName"))
    if pn:
        arts["pipe"].add(pn)

    return arts


def _artifacts_overlap(a: dict[str, set[str]], b: dict[str, set[str]]) -> bool:
    """두 artifact 집합 간 entity 공유 여부.
    cross-field 매칭 포함: file_written ↔ image_run (drop-then-execute).
    """
    # 같은 타입 공유
    for key in ("registry", "network", "pipe"):
        if a[key] & b[key]:
            return True
    # 같은 파일 touch
    if a["file_written"] & b["file_written"]:
        return True
    # drop-then-execute (양방향)
    if a["file_written"] & b["image_run"]:
        return True
    if a["image_run"] & b["file_written"]:
        return True
    return False


def _match_filter(row: pd.Series, filters: dict) -> bool:
    """
    Rule의 filters 블록 하나를 row에 적용 (AND).
    지원 연산자: contains, contains_any, endswith, in, not_contains, equals
    """
    for field, cond in filters.items():
        if field in ("note", "direction_note"):
            continue

        val = _str_val(row.get(field))

        if isinstance(cond, str):
            if val is None or cond.lower() not in val:
                return False

        elif isinstance(cond, dict):
            op = list(cond.keys())[0]
            arg = cond[op]

            if op in ("contains", "contains_any"):
                keywords = [arg] if isinstance(arg, str) else arg
                if val is None or not any(k.lower() in val for k in keywords):
                    return False

            elif op == "endswith":
                suffixes = [arg] if isinstance(arg, str) else arg
                if val is None or not any(val.endswith(s.lower()) for s in suffixes):
                    return False

            elif op == "in":
                items = [str(i).lower() for i in (arg if isinstance(arg, list) else [arg])]
                if val is None or val not in items:
                    return False

            elif op == "not_contains":
                keywords = [arg] if isinstance(arg, str) else arg
                if val is not None and any(k.lower() in val for k in keywords):
                    return False

            elif op == "equals":
                # 동적 참조(e.g. "Anchor.ProcessGuid")는 런타임 처리 — skip
                pass

        elif isinstance(cond, list):
            if val is None or not any(str(c).lower() in val for c in cond):
                return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# 2. Rule 파싱
# ──────────────────────────────────────────────────────────────────────────────
def _parse_anchor_specs(rule: dict) -> list[dict]:
    """Rule Anchor Event → [{'eids':[...], 'filters':{...}}, ...]"""
    anchor_block = rule["log_sources"]["Anchor Event"]
    channel      = anchor_block.get("channel", {})
    top_filters  = anchor_block.get("filters", {})
    conditions   = anchor_block.get("conditions", [])

    specs = []
    if conditions:
        for cond in conditions:
            eids = [int(e) for e in cond.get("channel", channel).get("event_codes", [])]
            merged = {**top_filters, **cond.get("filters", {})}
            specs.append({"eids": eids, "filters": merged})
    else:
        eids = [int(e) for e in channel.get("event_codes", [])]
        specs.append({"eids": eids, "filters": top_filters})

    return specs


def _parse_supporting_eids(rule: dict) -> list[int]:
    eids = []
    for s in rule["log_sources"].get("Supporting Events", []):
        eids.extend(int(e) for e in s.get("channel", {}).get("event_codes", []))
    return list(set(eids))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Anchor 탐색
# ──────────────────────────────────────────────────────────────────────────────

# Windows 시스템 프로세스 — 정당한 공격 행위가 이 이미지를 직접 anchor로 찍는 경우는 드묾.
# 프로세스 injection / LSASS 접근 등은 이 필터가 적용되기 전에 이미 별도 룰이 잡는다.
# 공격자가 living-off-the-land(rundll32, mshta, regsvr32 등)를 쓰는 경우엔 제외 대상 아님.
_NOISE_ANCHOR_IMAGES = {
    "svchost.exe", "lsass.exe", "wmiprvse.exe", "backgroundtaskhost.exe",
    "msmpeng.exe", "mpcmdrun.exe", "services.exe", "csrss.exe", "smss.exe",
    "dllhost.exe", "taskhostw.exe", "searchindexer.exe", "searchprotocolhost.exe",
    "searchfilterhost.exe", "wininit.exe", "winlogon.exe", "sppsvc.exe",
    "fontdrvhost.exe", "wsqmcons.exe", "logonui.exe", "sihost.exe",
    "audiodg.exe", "consent.exe", "conhost.exe", "runtimebroker.exe",
    "applicationframehost.exe", "startmenuexperiencehost.exe",
    "shellexperiencehost.exe", "cortana.exe", "lockapp.exe",
    "securityhealthservice.exe", "securityhealthsystray.exe",
    "compattelrunner.exe", "trustedinstaller.exe", "tiworker.exe",
    "usoclient.exe", "siemensindustrialdcs.exe", "defender.exe",
    # "System"은 EID 3 network anchor 에서만 노이즈 — 여기서 같이 처리.
    "system",
}


def _basename_lower(path) -> str:
    if not isinstance(path, str) or not path:
        return ""
    p = path.strip().lower()
    for sep in ("\\", "/"):
        if sep in p:
            p = p.rsplit(sep, 1)[-1]
    return p


def _is_noise_anchor(row: pd.Series) -> bool:
    """anchor row가 시스템 프로세스에서 발생한 노이즈인지 판정.

    - EID 1 (process create): NewProcessName / Image 기준
    - EID 3 (network): Image 기준 (svchost/System 네트워크 노이즈 필터)
    - EID 13/14 (registry): Image (레지스트리 수정한 프로세스) 기준
    - EID 10 (process access): SourceImage 기준
    """
    for fld in ("Image", "NewProcessName", "SourceImage", "Application"):
        img = _basename_lower(row.get(fld))
        if img and img in _NOISE_ANCHOR_IMAGES:
            return True
    return False


def _find_anchors(df: pd.DataFrame, rule: dict) -> pd.Index:
    """Rule Anchor 스펙 일치 인덱스 (OR 합집합). 노이즈 프로세스 anchor 제외."""
    specs = _parse_anchor_specs(rule)
    all_eids = [e for spec in specs for e in spec["eids"]]

    candidates = df[df["EventID"].isin(all_eids)]
    if candidates.empty:
        return pd.Index([])

    matched = set()
    for spec in specs:
        spec_eids = set(spec["eids"])
        spec_filt = spec["filters"]
        for idx, row in candidates[candidates["EventID"].isin(spec_eids)].iterrows():
            if spec_filt and not _match_filter(row, spec_filt):
                continue
            if _is_noise_anchor(row):
                continue
            matched.add(idx)
    return pd.Index(sorted(matched))


# ──────────────────────────────────────────────────────────────────────────────
# 4. ProcessGuid 그래프
# ──────────────────────────────────────────────────────────────────────────────
def _build_guid_graph(df: pd.DataFrame):
    """ParentProcessGuid → ProcessGuid 방향 그래프 + guid→row idx 맵."""
    children: dict[str, list[str]] = defaultdict(list)
    parents:  dict[str, list[str]] = defaultdict(list)
    guid_to_idxs: dict[str, list[int]] = defaultdict(list)

    for idx, row in df.iterrows():
        guid = _str_val(row.get("ProcessGuid"))
        if not guid:
            continue
        guid_to_idxs[guid].append(idx)

        parent_guid = _str_val(row.get("ParentProcessGuid"))
        if parent_guid:
            children[parent_guid].append(guid)
            parents[guid].append(parent_guid)

    return dict(children), dict(parents), dict(guid_to_idxs)


def _bfs_guids(start_guid: str, children: dict, parents: dict,
               hop_up: int, hop_down: int) -> set[str]:
    """start_guid 기준 위 hop_up / 아래 hop_down 방문 집합."""
    visited = {start_guid}

    frontier = {start_guid}
    for _ in range(hop_up):
        nxt = set()
        for g in frontier:
            for pg in parents.get(g, []):
                if pg not in visited:
                    visited.add(pg); nxt.add(pg)
        frontier = nxt
        if not frontier:
            break

    frontier = {start_guid}
    for _ in range(hop_down):
        nxt = set()
        for g in frontier:
            for cg in children.get(g, []):
                if cg not in visited:
                    visited.add(cg); nxt.add(cg)
        frontier = nxt
        if not frontier:
            break
    return visited


# ──────────────────────────────────────────────────────────────────────────────
# 5. Supporting / 필터 / 신뢰도
# ──────────────────────────────────────────────────────────────────────────────
def _collect_supporting(df: pd.DataFrame, anchor_idx: int,
                        supporting_eids: list[int], core_idxs: set[int],
                        before_sec: int, after_sec: int) -> list[int]:
    """Anchor ±time_window 내 Supporting EID 수집 (core 제외)."""
    if not supporting_eids:
        return []

    anchor = df.loc[anchor_idx]
    t0, host = anchor["TimeCreated"], anchor["Hostname"]
    t_start = t0 - timedelta(seconds=before_sec)
    t_end   = t0 + timedelta(seconds=after_sec)

    mask = (
        (df["Hostname"] == host) &
        (df["TimeCreated"] >= t_start) &
        (df["TimeCreated"] <= t_end) &
        (df["EventID"].isin(supporting_eids))
    )
    idxs = df[mask].index.tolist()
    return [i for i in idxs if i not in core_idxs]


def _apply_rule_filters(df: pd.DataFrame, anchor_idx: int, rule: dict) -> bool:
    """Anchor conditions 중 하나라도 통과하면 True."""
    specs = _parse_anchor_specs(rule)
    row = df.loc[anchor_idx]
    eid = int(row["EventID"])
    for spec in specs:
        if eid not in spec["eids"]:
            continue
        if not spec["filters"] or _match_filter(row, spec["filters"]):
            return True
    return False


def _calc_confidence(df: pd.DataFrame, all_idxs: list[int],
                     supporting_eids: list[int]) -> float:
    if not supporting_eids:
        return 1.0
    found = set(int(e) for e in df.loc[all_idxs, "EventID"].tolist())
    hit = len(set(supporting_eids) & found)
    return round(hit / len(set(supporting_eids)), 2)


# ──────────────────────────────────────────────────────────────────────────────
# 6. 공개 API
# ──────────────────────────────────────────────────────────────────────────────
def load_rules(rule_folder) -> list[dict]:
    """폴더 내 *.json 룰 파일 모두 로드."""
    folder = Path(rule_folder)
    rule_list = []
    for path in folder.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            rule_list.append(json.load(f))
    return rule_list


def run_grouping(df: pd.DataFrame,
                 rule_list: list[dict],
                 before_sec: int = 20,
                 after_sec:  int = 40,
                 hop_up:     int = 2,
                 hop_down:   int = 3,
                 apply_filters: bool = True,
                 use_shared_entity: bool = True,
                 max_anchors_per_rule: int = 80) -> list[dict]:
    """Rule 목록 기반 이벤트 그룹핑. max_anchors_per_rule로 폭발 방지."""
    print("ProcessGuid 그래프 구축 중...")
    children, parents, guid_to_idxs = _build_guid_graph(df)
    print(f"  그래프 노드 수: {len(guid_to_idxs):,}개 고유 ProcessGuid")

    # Artifact 사전 계산 — shared_entity 확장에서 중복 호출 제거 (anchor × window 이벤트 조합마다
    # 같은 row가 반복 조회되던 O(N²) 패턴 방지).
    print(f"Artifact 캐시 구축 중... ({len(df):,} 행)")
    artifacts_by_idx: dict[int, dict[str, set[str]]] = {}
    if use_shared_entity:
        for idx, row in df.iterrows():
            arts = _collect_artifacts(row)
            if any(arts.values()):
                artifacts_by_idx[idx] = arts
        print(f"  artifact 보유 이벤트: {len(artifacts_by_idx):,}개")

    groups: list[dict] = []

    for rule in rule_list:
        tid  = rule["technique"]["id"]
        name = rule["technique"]["name"]

        anchor_idxs = _find_anchors(df, rule)
        if anchor_idxs.empty:
            print(f"[SKIP] {tid}: Anchor 없음")
            continue

        # per-rule anchor 상한 (0 = 비활성). 0이면 모든 anchor 유지.
        if max_anchors_per_rule > 0 and len(anchor_idxs) > max_anchors_per_rule:
            orig = len(anchor_idxs)
            step = max(1, orig // max_anchors_per_rule)
            anchor_idxs = anchor_idxs[::step][:max_anchors_per_rule]
            print(f"[{tid}] Anchor {orig} → {len(anchor_idxs)}개 샘플링 (cap={max_anchors_per_rule})")

        supporting_eids = _parse_supporting_eids(rule)
        print(f"[{tid}] Anchor {len(anchor_idxs)}개  |  Supporting EID: {supporting_eids}")

        for anchor_idx in anchor_idxs:
            anchor_row  = df.loc[anchor_idx]
            anchor_guid = _str_val(anchor_row.get("ProcessGuid"))

            filter_passed = True
            if apply_filters:
                filter_passed = _apply_rule_filters(df, anchor_idx, rule)

            core_idxs: set[int] = {anchor_idx}

            if anchor_guid:
                reachable = _bfs_guids(anchor_guid, children, parents, hop_up, hop_down)
                for g in reachable:
                    core_idxs.update(guid_to_idxs.get(g, []))

                if "SourceProcessGUID" in df.columns:
                    eid10 = df[
                        (df["EventID"] == 10) &
                        (df["SourceProcessGUID"] == anchor_guid)
                    ]
                    core_idxs.update(eid10.index.tolist())

            t0, host = anchor_row["TimeCreated"], anchor_row["Hostname"]
            t_start = t0 - timedelta(seconds=before_sec)
            t_end   = t0 + timedelta(seconds=after_sec)

            core_idxs = {
                i for i in core_idxs
                if (df.loc[i, "Hostname"] == host and
                    t_start <= df.loc[i, "TimeCreated"] <= t_end)
            }

            # Shared Entity 확장 — 윈도우 내 anchor와 artifact 공유하는 이벤트 추가
            if use_shared_entity:
                anchor_arts = artifacts_by_idx.get(anchor_idx) or _collect_artifacts(anchor_row)
                if any(anchor_arts.values()):
                    window_mask = (
                        (df["Hostname"] == host) &
                        (df["TimeCreated"] >= t_start) &
                        (df["TimeCreated"] <= t_end)
                    )
                    for idx in df[window_mask].index:
                        if idx in core_idxs:
                            continue
                        event_arts = artifacts_by_idx.get(idx)
                        if event_arts is None:  # artifact 없는 이벤트는 캐시에서 누락됨
                            continue
                        if _artifacts_overlap(anchor_arts, event_arts):
                            core_idxs.add(idx)

            sup_idxs = _collect_supporting(df, anchor_idx, supporting_eids,
                                            core_idxs, before_sec, after_sec)
            all_idxs = sorted(core_idxs | set(sup_idxs))

            found_eids = sorted(set(int(e) for e in df.loc[all_idxs, "EventID"].tolist()))
            confidence = _calc_confidence(df, all_idxs, supporting_eids)

            groups.append({
                "group_id":        f"{tid.replace('.', '_')}_{anchor_idx}",
                "technique_id":    tid,
                "technique_name":  name,
                "anchor_idx":      anchor_idx,
                "anchor_eid":      int(anchor_row["EventID"]),
                "core_idxs":       sorted(core_idxs),
                "supporting_idxs": sup_idxs,
                "all_idxs":        all_idxs,
                "supporting_def":  supporting_eids,
                "supporting_hit":  found_eids,
                "confidence":      confidence,
                "filter_passed":   filter_passed,
            })

    total    = len(groups)
    filtered = sum(1 for g in groups if g["filter_passed"])
    print(f"\n총 {total}개 그룹 생성  |  필터 통과: {filtered}개")
    return groups


def merge_same_anchor(groups: list[dict]) -> list[dict]:
    """
    같은 anchor_idx를 가진 그룹을 병합 — 서로 다른 룰이 같은 이벤트를 anchor로
    지목한 경우 중복 제거. matched_techniques 필드에 모든 매칭 룰 id를 기록.
    """
    by_anchor: dict[int, list[dict]] = defaultdict(list)
    for g in groups:
        by_anchor[g["anchor_idx"]].append(g)

    merged: list[dict] = []
    for anchor_idx, glist in by_anchor.items():
        if len(glist) == 1:
            merged.append({**glist[0], "matched_techniques": [glist[0]["technique_id"]]})
            continue

        # 대표: 가장 높은 confidence. technique_id는 대표의 것 유지 (downstream 호환).
        rep = max(glist, key=lambda g: g.get("confidence", 0))
        all_tids = sorted({g["technique_id"] for g in glist})
        all_core = sorted(set(i for g in glist for i in g["core_idxs"]))
        all_sup  = sorted(set(i for g in glist for i in g["supporting_idxs"]))
        all_idxs = sorted(set(all_core) | set(all_sup))
        all_sup_def = sorted(set(e for g in glist for e in g.get("supporting_def", [])))

        merged.append({
            **rep,
            "core_idxs":          all_core,
            "supporting_idxs":    all_sup,
            "all_idxs":           all_idxs,
            "supporting_def":     all_sup_def,
            "matched_techniques": all_tids,
            "merged_anchor_count": len(glist),
        })
    return merged


def merge_shared_supporting(groups: list[dict], df: pd.DataFrame,
                            overlap_threshold: float = 1.0) -> list[dict]:
    """
    같은 technique_id 내에서 supporting_idxs가 overlap_threshold 이상
    겹치는 그룹을 Union-Find로 병합.
    """
    by_tech: dict[str, list[dict]] = defaultdict(list)
    for g in groups:
        by_tech[g["technique_id"]].append(g)

    merged_all: list[dict] = []
    for tid, tech_groups in by_tech.items():
        n = len(tech_groups)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                s_i = set(tech_groups[i]["supporting_idxs"])
                s_j = set(tech_groups[j]["supporting_idxs"])
                if not s_i and not s_j:
                    continue
                jacc = len(s_i & s_j) / len(s_i | s_j)
                if jacc >= overlap_threshold:
                    parent[find(i)] = find(j)

        clusters: dict[int, list[dict]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(tech_groups[i])

        for cluster in clusters.values():
            if len(cluster) == 1:
                merged_all.append(cluster[0])
                continue

            rep = min(cluster, key=lambda g: df.loc[g["anchor_idx"], "TimeCreated"])
            all_core = sorted(set(i for g in cluster for i in g["core_idxs"]))
            all_sup  = sorted(set(i for g in cluster for i in g["supporting_idxs"]))
            all_idxs = sorted(set(all_core) | set(all_sup))
            merged_all.append({
                **rep,
                "anchor_idxs":     sorted(g["anchor_idx"] for g in cluster),
                "core_idxs":       all_core,
                "supporting_idxs": all_sup,
                "all_idxs":        all_idxs,
                "supporting_hit":  sorted(set(int(e) for e in df.loc[all_idxs, "EventID"])),
                "confidence":      _calc_confidence(df, all_idxs, rep["supporting_def"]),
                "filter_passed":   all(g["filter_passed"] for g in cluster),
                "merged_count":    len(cluster),
            })

    return merged_all


def print_groups(groups: list[dict], show_filtered: bool = False) -> None:
    for g in groups:
        if not show_filtered and not g["filter_passed"]:
            continue
        status = "✓" if g["filter_passed"] else "✗(filtered)"
        print(f"[{g['group_id']}]  {status}")
        print(f"  Anchor      : EID {g['anchor_eid']}  (idx={g['anchor_idx']})")
        print(f"  Core        : {len(g['core_idxs'])}개  {g['core_idxs']}")
        print(f"  Supporting  : {len(g['supporting_idxs'])}개  {g['supporting_idxs']}")
        print(f"  All         : {len(g['all_idxs'])}개")
        print(f"  Support hit : {g['supporting_hit']}  / def: {g['supporting_def']}")
        print(f"  Confidence  : {g['confidence']}")
        print()
