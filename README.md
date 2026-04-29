# SCOPE — Artifact

This repository accompanies the CCS 2026 submission *SCOPE: Rethinking Attack Analysis as Uncertainty-Aware Security Reasoning under Incomplete Observations* and contains the complete pipeline, four reproduced baselines, the on-disk LLM description cache, and pre-computed evaluation outputs sufficient to reproduce every number reported in the paper without making any LLM API call.

---

## 1. What is here

```
Final_Code/
├── pipeline/           # Core SCOPE: data loader, behavior grouping, feature
│                       # extraction, distributional TTP mapping, multi-axis
│                       # transition scoring, Top-K Viterbi with hole-bridging.
├── experiments/        # Evaluation harness, four baseline re-implementations,
│   ├── baselines/      # (sigma, magic, shield, ttp_sequence/DeepAG)
│   ├── attack_flows.py # Reference attack flow per scenario (the GT).
│   ├── chain_align.py  # Chain-LCS scorer (plausibility-based).
│   └── ...
├── Technique Rule/     # 46 anchor-rule templates used by behavior grouping.
├── TTP_Data/           # MITRE ATT&CK Enterprise CSV + 53-campaign novelty
│                       # library (public).
├── output/             # Pre-computed per-scenario JSONs and aggregate metrics.
│   ├── _cache/         # Gemini-2.5-Flash description cache (32 MB).
│   ├── atomic/<cat>/<scenario>/   *_annotation.json, *_ttp_mapping.json,
│                                  *_feature_result.json, *_viterbi.json
│   ├── compound/<cat>/<scenario>/ same.
│   ├── baselines/      # Per-baseline result.json + _scores.json.
│   ├── eval_v2_results.json       # Aggregate numbers behind Table 2.
│   ├── _strict_metrics.json       # Strict-metric supplement (App J).
│   ├── _q5_scope_timings.json     # Q5 efficiency raw timings.
│   └── _robustness_scores.json    # Q2 robustness aggregate.
├── config.py           # Paths, hyperparameters, semantic backends.
├── main.py             # End-to-end SCOPE pipeline driver.
└── scripts/            # setup_data.sh, run_all.sh.
```

Two large public datasets are **not** included in this repository and are
fetched by `scripts/setup_data.sh` at install time:

- **OTRF Security-Datasets (Mordor)** — 35 attack scenarios, ~800 MB
  ([github.com/OTRF/Security-Datasets](https://github.com/OTRF/Security-Datasets)).
- **SigmaHQ rules** — ~2,140 Windows rules, ~40 MB
  ([github.com/SigmaHQ/sigma](https://github.com/SigmaHQ/sigma)).

---

## 2. Install

Tested on Python 3.11, Windows 11, and Ubuntu 22.04.

```bash
git clone <this-repo-url> SCOPE
cd SCOPE/Final_Code

python3 -m venv .venv
source .venv/bin/activate              # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Fetch the two public datasets we don't bundle (~840 MB total).
bash scripts/setup_data.sh
```

The on-disk LLM description cache ships with this repository under
`output/_cache/`, so **no Gemini API key is required to reproduce the
reported numbers**. Set `GEMINI_API_KEY` only if you intend to re-run
the description-generation step from scratch:

```bash
export GEMINI_API_KEY=...        # optional
```

---

## 3. Reproduce the paper

The fastest path to verify each table/figure:

```bash
# Q1 main comparison (Table 2)
python -m experiments.run_eval_v2

# Q2 robustness (Figure 3)
python experiments/_robustness_run.py            # uses cached LLM responses

# Q3 novelty coverage (Table 4)
python experiments/_novelty_score.py

# Q4 ablation (Figure 4)
python experiments/_ablation_run.py

# Q5 efficiency (§7.7 numbers)
python experiments/_q5_scope_timing.py
python experiments/_q5_collect_shield_timings.py

# Strict-metric supplement (App J)
python experiments/_strict_metrics.py

# Case study (§7.6, msf_record_mic / empire_mimikatz_logonpasswords)
python experiments/_case_study.py
```

Each script prints a per-scenario table and a macro-averaged summary,
and writes a JSON summary under `output/`.

To reproduce **all** of the above plus generate the paper figures in one
shot:

```bash
bash scripts/run_all.sh
```

Expected runtime on a single CPU-only workstation (Intel i7-12700F, 32 GB
RAM, no GPU): under 30 minutes for the full sweep (cache hot).

---

## 4. Re-run the SCOPE pipeline from scratch (optional)

If you want to regenerate the per-scenario `*_ttp_mapping.json` files
yourself instead of using the shipped pre-computed ones, do:

```bash
# Set your Gemini API key (description generation is the only LLM step)
export GEMINI_API_KEY=...

# Run end-to-end on every scenario in Dataset/
python main.py
```

This writes a fresh `output/<category>/<scenario>/` directory per
scenario. With a populated `output/_cache/` (shipped) the LLM step is
free; with a cold cache the full re-run costs roughly 1,500--2,000 API
calls in total (one per behavior group).

---

## 5. Reproducing each headline number

| Paper claim                              | Script                                            | Source file                  |
|------------------------------------------|---------------------------------------------------|------------------------------|
| Top-5 hit rate 0.84 (Table 2)             | `experiments/run_eval_v2.py`                     | `output/eval_v2_results.json`|
| Tech-LCS 0.68, Tac-LCS 0.77               | same                                              | same                         |
| 0.44 tech-LCS at 50% drop (Figure 3)      | `experiments/_robustness_run.py`                 | `output/_robustness_scores.json` |
| Chain-novel 0.91 (Table 4)                | `experiments/_novelty_score.py`                  | `output/_novelty_scores.json` |
| Behavior grouping −13 pp (Figure 4)       | `experiments/_ablation_run.py`                   | `output/_ablation/`           |
| 380 events/sec (§7.7)                     | `experiments/_q5_scope_timing.py`                | `output/_q5_scope_timings.json` |
| False-chain rate 0.057 (Table J.1)        | `experiments/_strict_metrics.py`                 | `output/_strict_metrics.json` |
| Hyperparameter ±0.014 (Figure 7)          | `experiments/_v20_XZ_sweep.py`, `_v22_*`         | `output/v20_XZ_sweep_results.json`, `v22_alpha_bypass_sweep_results.json` |

---

## 6. Anonymity statement

This artifact is anonymized for double-blind review. No author names,
email addresses, or institution-identifying paths appear in code or
configuration. The Gemini API key has been stripped from `config.py`;
set `GEMINI_API_KEY` via your shell environment if you wish to
regenerate the description cache.

---

## 7. License

Code: MIT (see `LICENSE`).
Data: OTRF Security-Datasets retains its original license; SigmaHQ
rules retain their original DRL 1.1 license. The MITRE ATT&CK CSV
under `TTP_Data/` is reproduced from the public ATT&CK STIX bundle and
is © MITRE Corporation.
