"""
SHIELD §3 -- Deviation Analyzer.

Implements the Local Outlier Factor (LOF) anomaly detector that flags
behavioral deviations in raw system logs, then extracts each anomalous
process together with its 1-hop ancestors and descendants (per Eq. 2 in
Gandhi et al., SHIELD 2025).

Adaptations for OTRF/Mordor:
- Original SHIELD log tuple is (process_id, process_name, event_type,
  object_id, object_data, timestamp). We project each row in the
  load_and_normalize DataFrame to a comparable tuple using ProcessGuid
  (process), Image (process_name), EventID (event_type), and a derived
  object identifier per event family (TargetFilename / TargetObject /
  DestinationIp:Port / TargetImage etc.).
- LOF features are the *numerical encodings* of (process_id, event_type,
  object_id) per Eq. 1 -- categorical → integer codes via pandas factorize
  and StandardScaler.
- We keep contamination=0.1 and k=20 from §3 deviation analyzer (b).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Object identifier extraction per event family
# ---------------------------------------------------------------------------

def _object_id(row: pd.Series) -> str:
    """Per-event-family object identifier (`oi` in SHIELD's log tuple)."""
    eid = row.get("EventID")
    try:
        eid = int(eid) if eid is not None and not pd.isna(eid) else None
    except (TypeError, ValueError):
        eid = None

    def _s(v) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    if eid in (1, 4688):
        return _s(row.get("Image"))                    # process image as object
    if eid == 11:
        return _s(row.get("TargetFilename"))
    if eid in (2, 23, 26):
        return _s(row.get("TargetFilename"))
    if eid in (12, 13, 14):
        return _s(row.get("TargetObject"))
    if eid == 3:
        ip = _s(row.get("DestinationIp"))
        port = _s(row.get("DestinationPort"))
        host = _s(row.get("DestinationHostname"))
        return f"{host or ip}:{port}" if (host or ip) else ""
    if eid == 22:
        return _s(row.get("QueryName"))
    if eid == 7:
        return _s(row.get("ImageLoaded"))
    if eid == 10:
        return _s(row.get("TargetImage"))
    if eid == 8:
        return _s(row.get("TargetImage"))
    if eid in (17, 18):
        return _s(row.get("PipeName"))
    if eid in (4624, 4625, 4634, 4648, 4672):
        return _s(row.get("TargetUserName") or row.get("SubjectUserName"))
    if eid == 4663:
        return _s(row.get("ObjectName"))
    if eid == 4104:
        sb = _s(row.get("ScriptBlockText"))
        return sb[:120]                                  # truncate large scripts
    if eid == 7045:
        return _s(row.get("ServiceName"))
    return _s(row.get("Message", ""))[:120]


# ---------------------------------------------------------------------------
# LOF over (process_id, event_type, object_id) -- Eq. 1
# ---------------------------------------------------------------------------

# Event IDs that the deviation analyzer should consider -- SHIELD's LOF runs
# over system-call-equivalent events. We mirror Sysmon coverage of process /
# file / registry / network / module / pipe / explicit-creds-logon families.
_LOF_EIDS = {
    1, 4688,                       # process_creation
    2,                             # file_change
    3,                             # network_connection
    7,                             # image_load
    8,                             # create_remote_thread
    9,                             # raw_access_thread
    10,                            # process_access
    11,                            # file_event
    12, 13, 14,                    # registry_*
    15,                            # create_stream_hash
    17, 18,                        # pipe_*
    19, 20, 21,                    # wmi_*
    22,                            # dns_query
    23, 26,                        # file_delete
    25,                            # process_tampering
    4103, 4104,                    # PowerShell scriptblock / module
    4624, 4625, 4648,              # logon / explicit-creds
    4663,                          # object access
    4698,                          # scheduled task created
    7045,                          # service install
}


@dataclass
class DeviationResult:
    """Output of the deviation analyzer."""
    df_subset: pd.DataFrame                  # rows the LOF actually scored
    is_anomalous: np.ndarray                 # bool[len(df_subset)]
    anomalous_rows: pd.DataFrame             # df_subset[is_anomalous]
    log_tuples: pd.DataFrame                 # (process_id, name, event_type, object_id, ts)
    n_total: int                             # total events before filter
    n_scored: int                            # total events fed into LOF
    n_anomalous: int                         # |is_anomalous == True|


def run_lof(df: pd.DataFrame,
            n_neighbors: int = 20,
            contamination: float = 0.1) -> DeviationResult:
    """Apply LOF to (encoded process_id, event_type, object_id).

    Mirrors SHIELD §3 deviation analyzer (a)+(b). Removes duplicate
    (process_id, event_type, object_id) tuples first as SHIELD does
    ('removing duplicate entries to ensure that unique process-object
    interactions are analyzed').
    """
    n_total = len(df)
    if "EventID" in df.columns:
        df = df[df["EventID"].isin(_LOF_EIDS)].copy()
    else:
        df = df.copy()

    # Project each row to SHIELD's log tuple
    proc_id = df.get("ProcessGuid")
    if proc_id is None:
        proc_id = df.get("ProcessId")
    df["_proc_id"] = proc_id.astype(str) if proc_id is not None else ""
    df["_proc_name"] = df.get("Image", pd.Series([""] * len(df))).astype(str)
    df["_evt_type"] = df.get("EventID", pd.Series([0] * len(df))).astype(str)
    df["_obj_id"] = df.apply(_object_id, axis=1)

    if df.empty:
        return DeviationResult(
            df_subset=df, is_anomalous=np.zeros(0, dtype=bool),
            anomalous_rows=df.iloc[0:0], log_tuples=df.iloc[0:0],
            n_total=n_total, n_scored=0, n_anomalous=0,
        )

    # Deduplicate (process_id, event_type, object_id) for LOF input only;
    # we still propagate anomaly status back to ALL matching rows.
    key_cols = ["_proc_id", "_evt_type", "_obj_id"]
    df_unique = df.drop_duplicates(subset=key_cols).reset_index(drop=False)
    df_unique.rename(columns={"index": "_orig_idx"}, inplace=True)

    if len(df_unique) < n_neighbors + 1:
        # Too few events for LOF -- flag none, return.
        return DeviationResult(
            df_subset=df, is_anomalous=np.zeros(len(df), dtype=bool),
            anomalous_rows=df.iloc[0:0], log_tuples=df_unique[key_cols],
            n_total=n_total, n_scored=len(df_unique), n_anomalous=0,
        )

    # Numerical encoding via pandas factorize per column (categorical → int)
    enc = np.column_stack([
        pd.factorize(df_unique[c])[0].astype(np.float64) for c in key_cols
    ])
    enc = StandardScaler().fit_transform(enc)

    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(df_unique) - 1),
                             contamination=contamination)
    pred = lof.fit_predict(enc)              # -1 = outlier, +1 = inlier
    unique_anom = pred == -1

    # Map unique-row anomaly status back to all rows that share the key tuple
    anomalous_keys = set(
        zip(df_unique.loc[unique_anom, "_proc_id"],
            df_unique.loc[unique_anom, "_evt_type"],
            df_unique.loc[unique_anom, "_obj_id"])
    )
    is_anom_all = df.apply(
        lambda r: (r["_proc_id"], r["_evt_type"], r["_obj_id"]) in anomalous_keys,
        axis=1,
    ).to_numpy()

    return DeviationResult(
        df_subset=df,
        is_anomalous=is_anom_all,
        anomalous_rows=df.loc[is_anom_all].copy(),
        log_tuples=df_unique[key_cols + ["_proc_name", "_orig_idx"]],
        n_total=n_total,
        n_scored=len(df_unique),
        n_anomalous=int(unique_anom.sum()),
    )


# ---------------------------------------------------------------------------
# 1-hop lineage extraction per Eq. 2 -- anomalous process + its
# immediate ancestors and descendants
# ---------------------------------------------------------------------------

def expand_with_one_hop_lineage(
    full_df: pd.DataFrame,
    anomalous_rows: pd.DataFrame,
) -> pd.DataFrame:
    """For each anomalous process, return all events in `full_df` whose
    ProcessGuid is the anomalous process itself, its parent, or any of its
    direct children. This implements Eq. 2 (R = ∪ G_i)."""
    if anomalous_rows.empty or "ProcessGuid" not in full_df.columns:
        return anomalous_rows.iloc[0:0]

    anomalous_guids = set(anomalous_rows.get("ProcessGuid", pd.Series()).dropna()
                          .astype(str).tolist())

    parent_guids: set[str] = set()
    child_guids: set[str] = set()
    if "ParentProcessGuid" in full_df.columns:
        # Parents: ParentProcessGuid for any anomalous ProcessGuid
        parent_guids = set(full_df.loc[full_df["ProcessGuid"].astype(str)
                                         .isin(anomalous_guids),
                                         "ParentProcessGuid"].dropna()
                           .astype(str).tolist())
        # Children: any event whose ParentProcessGuid is an anomalous ProcessGuid
        child_guids = set(full_df.loc[full_df["ParentProcessGuid"].astype(str)
                                        .isin(anomalous_guids),
                                        "ProcessGuid"].dropna()
                          .astype(str).tolist())

    keep = anomalous_guids | parent_guids | child_guids
    keep.discard("")
    keep.discard("nan")
    if not keep:
        return anomalous_rows
    return full_df.loc[full_df["ProcessGuid"].astype(str).isin(keep)].copy()


def filter_to_anomalous_subgraph(df: pd.DataFrame,
                                 n_neighbors: int = 20,
                                 contamination: float = 0.1
                                 ) -> tuple[pd.DataFrame, DeviationResult]:
    """End-to-end deviation analyzer entry point.

    Returns (filtered_df, lof_stats):
      filtered_df : events covering anomalous processes + their 1-hop lineage
      lof_stats   : raw LOF outputs for inspection
    """
    lof = run_lof(df, n_neighbors=n_neighbors, contamination=contamination)
    expanded = expand_with_one_hop_lineage(df, lof.anomalous_rows)
    return expanded, lof
