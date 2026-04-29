"""
SHIELD §3 -- Graph Analyzer.

Implements the four steps of the graph analyzer (Gandhi et al., SHIELD 2025
§3): (a) initial-infection-point detection on socket nodes, (b) suspicious
tag propagation, (c) pruning of untagged nodes, (d) Louvain community
detection.

Adaptations for OTRF/Mordor:
- Initial infection points: SHIELD assumes external sockets. For Sysmon EID 3
  we treat **non-private** destination IPs as external. Since some OTRF
  scenarios are entirely local-only (cmd/ntdsutil/wevtutil), if no external
  sockets exist we fall back to anomalous *processes themselves* as the
  propagation seed -- this preserves the algorithm semantics ("information
  flow originates from suspicious endpoints") rather than producing an empty
  graph.
- Node types: process / file / registry / socket / pipe / module -- broader
  than the paper's process/file/socket triple to fit Windows Sysmon.
"""
from __future__ import annotations

import ipaddress
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import community as community_louvain    # python-louvain
import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _is_external_ip(ip: str) -> bool:
    if not ip or ip in ("nan", "None", "-"):
        return False
    try:
        ipo = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    return not (ipo.is_private or ipo.is_loopback or ipo.is_link_local
                or ipo.is_multicast or ipo.is_reserved)


def _str(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def _proc_node(guid: str, image: str = "") -> tuple[str, dict]:
    return f"P:{guid}", {"type": "process", "guid": guid, "image": image}


def _file_node(path: str) -> tuple[str, dict]:
    return f"F:{path}", {"type": "file", "path": path}


def _reg_node(key: str) -> tuple[str, dict]:
    return f"R:{key}", {"type": "registry", "key": key}


def _net_node(host: str, port: str = "") -> tuple[str, dict]:
    label = f"{host}:{port}" if port else host
    return f"N:{label}", {"type": "socket", "host": host, "port": port,
                           "external": _is_external_ip(host)}


def _module_node(image: str) -> tuple[str, dict]:
    return f"M:{image}", {"type": "module", "image": image}


def _pipe_node(name: str) -> tuple[str, dict]:
    return f"PIPE:{name}", {"type": "pipe", "name": name}


def build_provenance_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a typed MultiDiGraph from a normalized event DataFrame.

    Edges carry: ``event_type`` (verb), ``ts`` (timestamp), ``eid``
    (Sysmon/Security event id), and a few event-specific attributes.
    """
    g = nx.MultiDiGraph()

    if df.empty:
        return g

    for _, row in df.iterrows():
        try:
            eid = int(row.get("EventID")) if not pd.isna(row.get("EventID")) else None
        except (TypeError, ValueError):
            eid = None
        ts = _str(row.get("TimeCreated"))
        guid = _str(row.get("ProcessGuid"))
        image = _str(row.get("Image"))
        if not guid:
            continue

        # Always add the source process node
        pn, pa = _proc_node(guid, image)
        if not g.has_node(pn):
            g.add_node(pn, **pa)
        elif image and not g.nodes[pn].get("image"):
            g.nodes[pn]["image"] = image

        # parent → child fork edge
        parent_guid = _str(row.get("ParentProcessGuid"))
        if parent_guid and eid in (1, 4688):
            parent_image = _str(row.get("ParentImage"))
            ppn, ppa = _proc_node(parent_guid, parent_image)
            if not g.has_node(ppn):
                g.add_node(ppn, **ppa)
            elif parent_image and not g.nodes[ppn].get("image"):
                g.nodes[ppn]["image"] = parent_image
            g.add_edge(ppn, pn, event_type="fork", ts=ts, eid=eid,
                       cmdline=_str(row.get("CommandLine"))[:200])

        # File events
        if eid == 11:                       # file create
            tf = _str(row.get("TargetFilename"))
            if tf:
                fn, fa = _file_node(tf)
                if not g.has_node(fn): g.add_node(fn, **fa)
                g.add_edge(pn, fn, event_type="write", ts=ts, eid=eid)
        elif eid in (23, 26):               # file delete
            tf = _str(row.get("TargetFilename"))
            if tf:
                fn, fa = _file_node(tf)
                if not g.has_node(fn): g.add_node(fn, **fa)
                g.add_edge(pn, fn, event_type="delete", ts=ts, eid=eid)

        # Registry events
        elif eid in (12, 13, 14):
            rk = _str(row.get("TargetObject"))
            if rk:
                rn, ra = _reg_node(rk)
                if not g.has_node(rn): g.add_node(rn, **ra)
                verb = "reg_set" if eid == 13 else ("reg_create" if eid == 12 else "reg_rename")
                g.add_edge(pn, rn, event_type=verb, ts=ts, eid=eid,
                           details=_str(row.get("Details"))[:120])

        # Network connections
        elif eid == 3:
            host = _str(row.get("DestinationHostname")) or _str(row.get("DestinationIp"))
            port = _str(row.get("DestinationPort"))
            if host:
                nn, na = _net_node(host, port)
                if not g.has_node(nn): g.add_node(nn, **na)
                g.add_edge(pn, nn, event_type="connect", ts=ts, eid=eid)

        # DNS queries -- treat as socket-like
        elif eid == 22:
            host = _str(row.get("QueryName"))
            if host:
                nn, na = _net_node(host)
                if not g.has_node(nn): g.add_node(nn, **na)
                g.add_edge(pn, nn, event_type="dns", ts=ts, eid=eid)

        # Image load (DLL etc.)
        elif eid == 7:
            il = _str(row.get("ImageLoaded"))
            if il:
                mn, ma = _module_node(il)
                if not g.has_node(mn): g.add_node(mn, **ma)
                g.add_edge(pn, mn, event_type="load", ts=ts, eid=eid)

        # Process access (open handle on another process)
        elif eid == 10:
            ti = _str(row.get("TargetImage"))
            tg = _str(row.get("TargetProcessGuid"))
            if tg:
                tn, ta = _proc_node(tg, ti)
                if not g.has_node(tn): g.add_node(tn, **ta)
                g.add_edge(pn, tn, event_type="access", ts=ts, eid=eid,
                           granted_access=_str(row.get("GrantedAccess")))

        # CreateRemoteThread
        elif eid == 8:
            ti = _str(row.get("TargetImage"))
            tg = _str(row.get("TargetProcessGuid"))
            if tg:
                tn, ta = _proc_node(tg, ti)
                if not g.has_node(tn): g.add_node(tn, **ta)
                g.add_edge(pn, tn, event_type="inject", ts=ts, eid=eid)

        # Named pipes
        elif eid in (17, 18):
            pp = _str(row.get("PipeName"))
            if pp:
                pn2, pa2 = _pipe_node(pp)
                if not g.has_node(pn2): g.add_node(pn2, **pa2)
                verb = "pipe_create" if eid == 17 else "pipe_connect"
                g.add_edge(pn, pn2, event_type=verb, ts=ts, eid=eid)

    return g


# ---------------------------------------------------------------------------
# §3 Graph Analyzer (a) initial infection points -- sockets external to host
# ---------------------------------------------------------------------------

def initial_infection_points(g: nx.MultiDiGraph) -> set[str]:
    """All external socket nodes (set I in Eq. 3)."""
    out = set()
    for n, attrs in g.nodes(data=True):
        if attrs.get("type") == "socket" and attrs.get("external"):
            out.add(n)
    return out


# ---------------------------------------------------------------------------
# §3 Graph Analyzer (b) suspicious tag propagation
# ---------------------------------------------------------------------------

def propagate_tags(g: nx.MultiDiGraph, seeds: Iterable[str]) -> set[str]:
    """BFS that tags every node reachable from a seed, plus every node
    reachable in the *reverse* direction within one hop (the receiving end
    of a connection).

    SHIELD's relayData function is not a no-op for processes that consume
    socket data (every process is a potential relay), so we approximate by
    treating any process that has an in-edge from a tagged node as tagged,
    then propagating forward through its out-edges. We additionally include
    one-hop reverse traversal so that if a process *connects out* to an
    external socket, the process is tagged."""
    tagged: set[str] = set(seeds)
    if not tagged:
        return tagged
    frontier = set(tagged)
    while frontier:
        nxt: set[str] = set()
        for n in frontier:
            # successors (forward)
            for s in g.successors(n):
                if s not in tagged:
                    tagged.add(s); nxt.add(s)
            # predecessors (reverse one-hop)
            for p in g.predecessors(n):
                if p not in tagged:
                    tagged.add(p); nxt.add(p)
        frontier = nxt
    return tagged


# ---------------------------------------------------------------------------
# §3 Graph Analyzer (c) prune untagged nodes -- Eq. 4
# ---------------------------------------------------------------------------

def prune_untagged(g: nx.MultiDiGraph, tagged: set[str]) -> nx.MultiDiGraph:
    return g.subgraph(tagged).copy()


# ---------------------------------------------------------------------------
# §3 Graph Analyzer (d) Louvain community detection
# ---------------------------------------------------------------------------

def louvain_communities(g: nx.MultiDiGraph, seed: int = 0) -> dict[int, list[str]]:
    """Run Louvain on the *undirected* projection of g (python-louvain
    requires an undirected graph). Returns {community_id: [nodes]}."""
    if g.number_of_nodes() == 0:
        return {}

    ug = nx.Graph()
    for u, v, _ in g.edges(data=False, keys=True):
        if ug.has_edge(u, v):
            ug[u][v]["weight"] = ug[u][v].get("weight", 1) + 1
        else:
            ug.add_edge(u, v, weight=1)
    for n in g.nodes():
        ug.add_node(n)

    if ug.number_of_edges() == 0:
        # Each node is its own community
        return {i: [n] for i, n in enumerate(ug.nodes())}

    partition = community_louvain.best_partition(ug, random_state=seed)
    out: dict[int, list[str]] = defaultdict(list)
    for n, c in partition.items():
        out[c].append(n)
    return dict(out)


# ---------------------------------------------------------------------------
# Public entry point -- full graph-analyzer pipeline
# ---------------------------------------------------------------------------

@dataclass
class GraphAnalyzerResult:
    full_graph:    nx.MultiDiGraph
    seeds:         set[str]
    tagged:        set[str]
    reduced_graph: nx.MultiDiGraph
    communities:   dict[int, list[str]] = field(default_factory=dict)


def run_graph_analyzer(df: pd.DataFrame,
                       fallback_seeds: Iterable[str] | None = None,
                       ) -> GraphAnalyzerResult:
    """Build graph → propagate tags → prune → cluster.

    `fallback_seeds`: process node ids to use as propagation seeds when no
    external socket is present (e.g., LOF-flagged processes). Should be
    formatted as ``f"P:{ProcessGuid}"`` to match node naming.
    """
    g = build_provenance_graph(df)
    seeds = initial_infection_points(g)
    if not seeds and fallback_seeds:
        seeds = {s for s in fallback_seeds if g.has_node(s)}
    tagged = propagate_tags(g, seeds) if seeds else set(g.nodes())
    reduced = prune_untagged(g, tagged)
    comms = louvain_communities(reduced)
    return GraphAnalyzerResult(
        full_graph=g, seeds=seeds, tagged=tagged,
        reduced_graph=reduced, communities=comms,
    )
