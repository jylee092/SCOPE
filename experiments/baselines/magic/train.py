"""
Train MAGIC's GMAE on a benign-approximated pool of Mordor sub-graphs.

Mordor scenarios all contain attack activity, so we approximate the
"benign training pool" by taking each scenario's full provenance graph,
removing edges incident on processes the LOF deviation analyzer flagged
as anomalous, and pooling the remainders. This matches the SHIELD pipe-
line we already use as an upstream filter and corresponds to MAGIC's
assumption of "benign system entities and behaviours" for self-super-
vised pre-training.

Saves:
    output/baselines/magic/_model/model.pt
    output/baselines/magic/_model/benign_emb.pt   (KNN reference embeddings)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import config
from pipeline.data_loader import load_and_normalize
from experiments.baselines.shield.deviation import filter_to_anomalous_subgraph
from experiments.baselines.shield.graph import build_provenance_graph
from experiments.baselines.magic.graph_io import (
    nx_to_pyg, benign_subgraph, NODE_FEAT_DIM,
)
from experiments.baselines.magic.model import (
    GMAE, MagicConfig, save_model,
)


MODEL_DIR = config.OUTPUT_BASE_DIR / "baselines" / "magic" / "_model"


def _scenario_paths():
    return sorted(config.DATASET_FOLDER.rglob("*.json"))


def build_benign_graphs():
    """For each scenario: load → LOF → strip anomalous-process edges → return PyG graphs."""
    out = []
    for scen in _scenario_paths():
        try:
            df = load_and_normalize(str(scen))
            df = df.reset_index().rename(columns={"index": "_orig_idx"})
            _, lof = filter_to_anomalous_subgraph(df)
            anom_guids = set(
                lof.anomalous_rows.get("ProcessGuid", [])
                .dropna().astype(str).tolist()
            )
            g_full = build_provenance_graph(df)
            g_ben = benign_subgraph(g_full, anom_guids)
            data, _ = nx_to_pyg(g_ben)
            if data.x.size(0) >= 4 and data.edge_index.size(1) >= 4:
                out.append((scen.stem, data))
                print(f"  benign graph for {scen.stem}: "
                      f"{data.x.size(0)} nodes / {data.edge_index.size(1)} edges")
        except Exception as e:
            print(f"  [warn] {scen.stem}: {type(e).__name__}: {e}")
    return out


def train(epochs: int = 50,
          lr: float = 1e-3,
          weight_decay: float = 5e-4,
          seed: int = 0) -> None:
    torch.manual_seed(seed)
    print("[magic] building benign training graphs ...")
    graphs = build_benign_graphs()
    print(f"[magic] {len(graphs)} graphs ready")
    if not graphs:
        raise RuntimeError("no usable benign graphs")

    cfg = MagicConfig(in_dim=NODE_FEAT_DIM, e_dim=1)
    model = GMAE(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)

    print("[magic] training GMAE ...")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()
        idx = torch.randperm(len(graphs)).tolist()
        for i in idx:
            _, data = graphs[i]
            if data.x.size(0) < 4:
                continue
            optim.zero_grad()
            loss = model(data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item()
            n += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:>3}: avg_loss={running/max(n,1):.4f} "
                  f"({time.time()-t0:.1f}s, {n} graphs)")

    # Build benign reference embeddings (used as KNN baseline at test-time)
    model.eval()
    benign_emb_chunks = []
    with torch.no_grad():
        for _, data in graphs:
            emb = model.embed(data)                       # [N, H * n_layers]
            benign_emb_chunks.append(emb)
    benign_emb = torch.cat(benign_emb_chunks, dim=0)
    # Subsample to keep KNN fast
    if benign_emb.size(0) > 30_000:
        idx = torch.randperm(benign_emb.size(0))[:30_000]
        benign_emb = benign_emb[idx]
    print(f"[magic] benign reference embedding pool: {benign_emb.shape}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, MODEL_DIR / "model.pt")
    torch.save(benign_emb, MODEL_DIR / "benign_emb.pt")
    print(f"[magic] saved → {MODEL_DIR}")


if __name__ == "__main__":
    train()
