"""
Section 0. 데이터 전처리
 - JSONL 이벤트 로그 로드
 - Message 필드 파싱 → 구조화 컬럼 확장
 - ProcessId / LogonId / TimeCreated 정규화
"""
from __future__ import annotations

import json
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 1. 이벤트 로드
# ──────────────────────────────────────────────────────────────────────────────
_WINLOGBEAT_RENAME = {
    "event_id": "EventID",
    "source_name": "SourceName",
    "computer_name": "Hostname",
    "log_name": "Channel",
    "message": "Message",
    "record_number": "RecordNumber",
    "task": "Task",
    "opcode": "Opcode",
    "level": "Severity",
    "type": "EventType",
}


def _normalize_winlogbeat(obj: dict) -> dict:
    """winlogbeat 스키마(`event_id` + 중첩 `event_data`)를 nxlog flat 스키마로 평탄화.

    event_data 내부 필드(Image, ProcessId, TargetObject 등)를 top-level로 올리고,
    lowercase 공통 필드(event_id, source_name, ...)를 PascalCase로 rename.
    """
    if "event_id" not in obj or "EventID" in obj:
        return obj  # already nxlog-style or unrelated

    flat = {}
    ed = obj.get("event_data") or {}
    if isinstance(ed, dict):
        flat.update(ed)
    for k, v in obj.items():
        if k == "event_data":
            continue
        flat[_WINLOGBEAT_RENAME.get(k, k)] = v
    return flat


def load_events(file_path: str) -> pd.DataFrame:
    """JSONL 형식 이벤트 로그 → DataFrame."""
    events = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            events.append(_normalize_winlogbeat(obj))

    df = pd.DataFrame(events)
    if "TimeCreated" in df.columns:
        df["TimeCreated"] = pd.to_datetime(df["TimeCreated"])
        df = df.sort_values("TimeCreated").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Message 파싱
# ──────────────────────────────────────────────────────────────────────────────
def _scalar(v):
    """Series/list → 첫 스칼라. NaN/None → None."""
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) > 0 else None
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None:
        return None
    try:
        if isinstance(v, float) and pd.isna(v):
            return None
    except Exception:
        pass
    return v


def parse_message(message, row_data: dict) -> dict:
    """EventLog Message 필드를 description + additional_info dict로 분해."""
    if isinstance(message, pd.Series):
        message = message.iloc[0] if len(message) > 0 else None
    if message is None:
        return {"description": None, "additional_info": None}
    try:
        if isinstance(message, float) and pd.isna(message):
            return {"description": None, "additional_info": None}
    except Exception:
        pass

    lines = str(message).split("\n")
    description = lines[0].strip()

    parsed_data = {}
    for line in lines[1:]:
        if "=" in line or ":" in line:
            m = re.match(r"\s*(\w+)\s*[=:]\s*(.+)", line)
            if m:
                k, v = m.groups()
                parsed_data[k.strip()] = v.strip()

    row_values = set()
    for v in row_data.values():
        sv = _scalar(v)
        if sv is not None:
            row_values.add(str(sv))

    additional_info = {k: v for k, v in parsed_data.items() if v not in row_values}
    return {
        "description": description,
        "additional_info": additional_info if additional_info else None,
    }


def process_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Message 컬럼을 파싱하여 description으로 교체 + 추가 컬럼 확장."""
    df = df.reset_index(drop=True)

    print("Parsing messages...")
    parsed_results = []
    for idx, row in enumerate(df.itertuples(index=False)):
        row_dict = row._asdict()
        parsed_results.append(parse_message(row_dict.get("Message"), row_dict))
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} rows...")

    df["Message"] = [r["description"] for r in parsed_results]

    add_list = [r["additional_info"] if r["additional_info"] else {} for r in parsed_results]
    if any(add_list):
        add_df = pd.DataFrame(add_list, index=df.index)
        valid_cols = [c for c in add_df.columns if str(c).strip() and not str(c).isdigit()]
        add_df = add_df[valid_cols]

        common = set(df.columns) & set(add_df.columns)
        for col in common:
            df[col] = df[col].fillna(add_df[col])

        new_cols = [c for c in add_df.columns if c not in df.columns]
        if new_cols:
            df = pd.concat([df, add_df[new_cols]], axis=1)
            print(f"\nAdded {len(new_cols)} new columns from Message.")
            print(f"  {new_cols}")
    else:
        print("\nNo additional info found in messages.")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. 정규화
# ──────────────────────────────────────────────────────────────────────────────
def _to_int(val):
    """hex/decimal/float 문자열 → int. 실패 시 None."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s == "nan":
        return None
    try:
        return int(float(s)) if "." in s else int(s, 0)
    except (ValueError, TypeError):
        return None


def normalize_process_ids(df: pd.DataFrame) -> pd.DataFrame:
    """ProcessId 계열 필드를 nullable Int64로 _int suffix 컬럼에 저장."""
    pid_fields = [
        "ProcessId", "NewProcessId", "ParentProcessId",
        "SourceProcessId", "TargetProcessId",
    ]
    for field in pid_fields:
        if field in df.columns:
            df[f"{field}_int"] = pd.array(
                [_to_int(v) for v in df[field]],
                dtype=pd.Int64Dtype(),
            )
    return df


def _norm_logon_id_val(val):
    """단일 LogonId → 소문자 hex. 0x0/NaN → None."""
    if isinstance(val, pd.Series):
        val = val.iloc[0] if len(val) > 0 else None
    if val is None:
        return None
    try:
        if isinstance(val, float) and pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip().lower()
    if not s or s == "nan" or s == "0x0":
        return None
    return s


def normalize_logon_ids(df: pd.DataFrame) -> pd.DataFrame:
    """EID별 LogonId 필드를 norm_logon_id 컬럼으로 통합."""
    df["norm_logon_id"] = None

    if "LogonId" in df.columns:
        mask1 = df["EventID"] == 1
        if mask1.any():
            df.loc[mask1, "norm_logon_id"] = df.loc[mask1, "LogonId"].apply(_norm_logon_id_val)

    if "SubjectLogonId" in df.columns:
        mask_sec = df["EventID"].isin([4688, 4689])
        if mask_sec.any():
            df.loc[mask_sec, "norm_logon_id"] = (
                df.loc[mask_sec, "SubjectLogonId"].apply(_norm_logon_id_val)
            )
    return df


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    @timestamp > TimeCreated > EventTime 우선순위로 시간 컬럼 결정 후
    TimeCreated 컬럼명으로 통일, datetime(UTC)으로 변환 + 정렬.
    """
    if "@timestamp" in df.columns:
        target = "@timestamp"
    elif "TimeCreated" in df.columns:
        target = "TimeCreated"
    elif "EventTime" in df.columns:
        target = "EventTime"
    else:
        raise ValueError("시간 컬럼(@timestamp, TimeCreated, EventTime) 없음.")

    print(f"Using '{target}' as the primary timestamp.")
    df[target] = pd.to_datetime(df[target], format="mixed", utc=True)

    if target != "TimeCreated":
        if "TimeCreated" in df.columns:
            df["TimeCreated_Original"] = df["TimeCreated"]
        df["TimeCreated"] = df[target]

    return df.sort_values("TimeCreated").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 공개 API
# ──────────────────────────────────────────────────────────────────────────────
def load_and_normalize(file_path: str) -> pd.DataFrame:
    """JSONL 로드 → Message 파싱 → 정규화까지 일괄 실행."""
    df = load_events(file_path)
    print(f"총 이벤트 수: {len(df):,}")
    df = process_messages(df)
    df = normalize_timestamps(df)
    df = normalize_process_ids(df)
    df = normalize_logon_ids(df)
    return df
