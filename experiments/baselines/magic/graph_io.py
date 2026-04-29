"""
Convert SHIELD's NetworkX MultiDiGraph → PyG `Data` for MAGIC.

Node feature  = [type one-hot (6) | hashed identity (16) | type-specific flags (4)]
                identity hash gives per-process / per-file discriminative signal so
                the GAT autoencoder can learn meaningful KNN distances; pure type
                one-hot was too coarse — all processes ended up with identical
                embeddings, collapsing the anomaly-score distribution.
Edge feature  = one-hot of edge verb (used as initial weight only — basic
                GATConv does not consume edge attributes; the type information
                propagates via the structural masking objective).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import networkx as nx
import torch
from torch_geometric.data import Data


_NODE_TYPES = OrderedDict([
    ("process",  0),
    ("file",     1),
    ("registry", 2),
    ("socket",   3),
    ("module",   4),
    ("pipe",     5),
])
N_NODE_TYPES = len(_NODE_TYPES)


_EDGE_VERBS = OrderedDict([
    ("fork",          0),
    ("write",         1),
    ("delete",        2),
    ("reg_set",       3),
    ("reg_create",    4),
    ("reg_rename",    5),
    ("connect",       6),
    ("dns",           7),
    ("load",          8),
    ("access",        9),
    ("inject",       10),
    ("pipe_create",  11),
    ("pipe_connect", 12),
])
N_EDGE_TYPES = len(_EDGE_VERBS)


_HASH_DIM = 16
_FLAG_DIM = 4

# Total node feature dim (kept exposed so model.py can read)
NODE_FEAT_DIM = N_NODE_TYPES + _HASH_DIM + _FLAG_DIM


def _node_type_id(attrs: dict) -> int:
    t = (attrs.get("type") or "").lower()
    return _NODE_TYPES.get(t, 0)                       # default = process


def _edge_type_id(d: dict) -> int:
    v = (d.get("event_type") or "").lower()
    return _EDGE_VERBS.get(v, 0)


def _identity_string(attrs: dict, t: str) -> str:
    """Per-node identity string used to seed the hash feature."""
    if t == "process":
        img = (attrs.get("image") or "").lower()
        return img.rsplit("\\", 1)[-1]                 # image filename
    if t == "file":
        return (attrs.get("path") or "").lower()
    if t == "registry":
        return (attrs.get("key") or "").lower()
    if t == "socket":
        return f"{attrs.get('host','')}:{attrs.get('port','')}"
    if t == "module":
        img = (attrs.get("image") or "").lower()
        return img.rsplit("\\", 1)[-1]
    if t == "pipe":
        return (attrs.get("name") or "").lower()
    return ""


def _hash_vector(s: str, dim: int = _HASH_DIM) -> torch.Tensor:
    """Stable per-string deterministic hash → unit vector."""
    if not s:
        return torch.zeros(dim, dtype=torch.float32)
    h = hashlib.sha256(s.encode("utf-8")).digest()
    # take dim bytes, normalize to [-1, 1]
    raw = torch.tensor([b / 127.5 - 1.0 for b in h[:dim]],
                          dtype=torch.float32)
    return torch.nn.functional.normalize(raw, p=2, dim=0)


def _type_flags(attrs: dict, t: str) -> torch.Tensor:
    """Type-specific binary flags carrying coarse semantics."""
    f = torch.zeros(_FLAG_DIM, dtype=torch.float32)
    if t == "process":
        img = (attrs.get("image") or "").lower()
        # flag 0 — system process path
        if "\\windows\\" in img: f[0] = 1.0
        # flag 1 — known interpreter
        if any(x in img for x in ("powershell", "cmd.exe", "wscript",
                                       "cscript", "rundll32", "regsvr32")):
            f[1] = 1.0
    elif t == "file":
        path = (attrs.get("path") or "").lower()
        # flag 2 — sensitive directory (Temp, Public, AppData)
        if any(x in path for x in ("\\temp\\", "\\public\\",
                                       "\\appdata\\", "\\downloads\\")):
            f[2] = 1.0
    elif t == "socket":
        if attrs.get("external"):
            f[3] = 1.0
    return f


def nx_to_pyg(g: nx.MultiDiGraph) -> tuple[Data, list[str]]:
    """Returns (Data, node_id_list)."""
    nodes = list(g.nodes())
    idx_of = {n: i for i, n in enumerate(nodes)}

    n = len(nodes)
    x = torch.zeros(n, NODE_FEAT_DIM, dtype=torch.float32)
    for i, node in enumerate(nodes):
        attrs = g.nodes[node]
        t_str = (attrs.get("type") or "").lower()
        x[i, _node_type_id(attrs)] = 1.0
        # hash slice
        x[i, N_NODE_TYPES:N_NODE_TYPES + _HASH_DIM] = _hash_vector(
            _identity_string(attrs, t_str)
        )
        # flag slice
        x[i, N_NODE_TYPES + _HASH_DIM:] = _type_flags(attrs, t_str)

    edge_set: dict[tuple[int, int, int], None] = {}
    for u, v, d in g.edges(data=True):
        ui = idx_of.get(u); vi = idx_of.get(v)
        if ui is None or vi is None or ui == vi:
            continue
        edge_set[(ui, vi, _edge_type_id(d))] = None
    if edge_set:
        edges = list(edge_set.keys())
        ei = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]],
                            dtype=torch.long)
        ea = torch.tensor([e[2] for e in edges], dtype=torch.long)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, dtype=torch.long)

    return Data(x=x, edge_index=ei, edge_attr=ea), nodes


def benign_subgraph(g: nx.MultiDiGraph, anom_process_guids: set[str]) -> nx.MultiDiGraph:
    """Return a copy of `g` with the LOF-flagged anomalous process nodes
    REMOVED ENTIRELY (along with their incident edges). This matches MAGIC's
    "benign-only training" assumption — the anomalous processes' embeddings
    must not appear in the benign reference pool, otherwise the KNN distance
    at test time collapses for the very nodes we want to detect."""
    if not anom_process_guids:
        return g.copy()
    bad = {f"P:{guid}" for guid in anom_process_guids}
    h = nx.MultiDiGraph()
    for n, attrs in g.nodes(data=True):
        if n in bad:
            continue
        h.add_node(n, **attrs)
    for u, v, k, d in g.edges(keys=True, data=True):
        if u in bad or v in bad:
            continue
        if u in h and v in h:
            h.add_edge(u, v, key=k, **d)
    return h
