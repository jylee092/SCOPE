# SCOPE

SCOPE is an uncertainty-aware framework for reconstructing and interpreting
attack behaviors from incomplete endpoint logs. It (1) groups raw events into
behavior-centric units that tolerate missing or disordered events, (2) maps
each unit to a *distribution* over MITRE ATT&CK techniques instead of a single
label, and (3) runs a Top-K Viterbi sequence inference combining tactical,
semantic, and causal compatibility with explicit missing-step (skip) handling
to recover the overall attack chain. Structural inference is fully
deterministic; a large language model is used only to generate auxiliary
natural-language descriptions consumed by the technique-matching step.

This repository contains the complete pipeline, four reproduced baselines
(Sigma, MAGIC, SHIELD, DeepAG), the on-disk LLM description cache, and
pre-computed evaluation outputs, so the results can be regenerated without any
LLM API call.

---

## 1. Repository layout

```
Final_Code/
├── pipeline/           # Core SCOPE: data loader, behavior grouping, feature
│                       # extraction, distributional TTP mapping, multi-axis
│                       # transition scoring, Top-K Viterbi with hole-bridging.
├── experiments/        # Evaluation harness + baseline re-implementations.
│   ├── baselines/      # sigma, magic, shield, ttp_sequence (DeepAG).
│   ├── attack_flows.py # Reference attack flow per scenario (ground truth).
│   ├── chain_align.py  # Chain-LCS scorer (plausibility-based).
│   ├── ccs_revision/   # Additional evaluations (strict metrics, transition
│   │                   # matrix, supervised baseline, sub-module self-metrics,
│   │                   # unified baseline scoring, LLM-swap ablation).
│   └── ...
├── Technique Rule/     # Anchor-rule templates used by behavior grouping.
├── TTP_Data/           # MITRE ATT&CK Enterprise CSV + campaign library (public).
├── output/             # Pre-computed per-scenario JSONs and aggregate metrics.
│   ├── _cache/         # LLM description cache (no API key needed to reproduce).
│   ├── atomic/<cat>/<scenario>/   *_annotation.json, *_ttp_mapping.json,
│   │                              *_feature_result.json, *_viterbi.json
│   ├── compound/<cat>/<scenario>/ same.
│   ├── baselines/      # Per-baseline result.json + _scores.json.
│   └── _ccs_revision/  # Outputs of experiments/ccs_revision/.
├── config.py           # Paths, hyperparameters, semantic backends.
├── main.py             # End-to-end SCOPE pipeline driver.
└── scripts/            # setup_data.sh, run_all.sh.
```

Two large public datasets are **not** bundled and are fetched by
`scripts/setup_data.sh` at install time:

- **OTRF Security-Datasets (Mordor)** — attack scenarios, ~800 MB
  ([github.com/OTRF/Security-Datasets](https://github.com/OTRF/Security-Datasets)).
- **SigmaHQ rules** — Windows detection rules, ~40 MB
  ([github.com/SigmaHQ/sigma](https://github.com/SigmaHQ/sigma)).

---

## 2. Install

Tested on Python 3.11 (Windows 11 and Ubuntu 22.04).

```bash
git clone https://github.com/jylee092/SCOPE.git
cd SCOPE/Final_Code

python3 -m venv .venv
source .venv/bin/activate              # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Fetch the two public datasets not bundled here (~840 MB total).
bash scripts/setup_data.sh
```

The on-disk LLM description cache ships under `output/_cache/`, so **no API key
is required to reproduce the precomputed results**. Set `GEMINI_API_KEY` only if
you intend to re-run the description-generation step from scratch:

```bash
export GEMINI_API_KEY=...        # optional
```

---

## 3. Running the evaluations

Each script prints a per-scenario table plus a macro-averaged summary and writes
a JSON/CSV summary under `output/`.

```bash
# Main TTP/chain comparison vs baselines
python -m experiments.run_eval_v2

# Robustness under random log loss
python experiments/_robustness_run.py

# Novelty coverage (unseen attack compositions)
python experiments/_novelty_score.py

# Component ablation
python experiments/_ablation_run.py

# Efficiency / timing
python experiments/_q5_scope_timing.py

# Plausibility-based per-group evaluation (Hit@K)
python -m experiments.run_eval_plausible
```

Additional evaluations under `experiments/ccs_revision/` (run as modules):

```bash
python -m experiments.ccs_revision.r3_strict          # strict-metric re-scoring
python -m experiments.ccs_revision.r4_sensitivity     # transition matrix + alpha sensitivity
python -m experiments.ccs_revision.r6_supervised      # supervised-classifier baseline (LOSO)
python -m experiments.ccs_revision.r7_self_metrics    # per-submodule self-evaluation
python -m experiments.ccs_revision.r8_baseline_strict # strict/standard metrics, all methods
python -m experiments.ccs_revision.r9_unified         # unified comparison table
python -m experiments.ccs_revision.r10_template       # LLM-swap: template (no-LLM) variant
python -m experiments.ccs_revision.r10_gpt            # LLM-swap: GPT variant (needs OPENAI_API_KEY)
python -m experiments.ccs_revision.r10_eval           # LLM-swap comparison
```

To run the core evaluations plus figure rendering in one shot:

```bash
bash scripts/run_all.sh
```

Expected runtime on a CPU-only workstation (Intel i7-12700F, 32 GB RAM, no GPU):
under ~30 minutes for the full core sweep with a warm cache.

---

## 4. Re-running the pipeline from scratch (optional)

To regenerate the per-scenario `*_ttp_mapping.json` files instead of using the
shipped pre-computed ones:

```bash
export GEMINI_API_KEY=...   # description generation is the only LLM step
python main.py              # runs end-to-end on every scenario in Dataset/
```

This writes a fresh `output/<category>/<scenario>/` per scenario. With the
shipped `output/_cache/` the LLM step is free; with a cold cache a full re-run
issues roughly one API call per behavior group.

---

## 5. License

Code: MIT (see `LICENSE`). Data: OTRF Security-Datasets retains its original
license; SigmaHQ rules retain their original DRL 1.1 license. The MITRE ATT&CK
CSV under `TTP_Data/` is reproduced from the public ATT&CK STIX bundle and is
© MITRE Corporation.
