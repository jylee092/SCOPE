#!/usr/bin/env bash
# =============================================================
# SCOPE artifact -- end-to-end reproduction.
#
# Runs every evaluation script that backs a number, table, or
# figure in the paper, using the shipped LLM cache. Total wall
# time on i7-12700F (CPU-only) is roughly 25-30 minutes.
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d "Dataset" ] || [ ! -d "_sigma_rules" ]; then
  echo "[error] Dataset/ or _sigma_rules/ missing. Run scripts/setup_data.sh first."
  exit 1
fi

run() {
  echo
  echo "═══════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "═══════════════════════════════════════════════════════════════"
  shift
  "$@"
}

run "Q1 main comparison (Table 2)"            python -m experiments.run_eval_v2
run "Q2 robustness (Figure 3)"                python  experiments/_robustness_run.py
run "Q3 novelty coverage (Table 4)"           python  experiments/_novelty_score.py
run "Q4 ablation (Figure 4)"                  python  experiments/_ablation_run.py
run "Q5 SCOPE timing (§7.7)"                  python  experiments/_q5_scope_timing.py
run "Q5 SHIELD timing (§7.7)"                 python  experiments/_q5_collect_shield_timings.py
run "Strict-metric supplement (App J)"        python  experiments/_strict_metrics.py
run "Case study (§7.6)"                       python  experiments/_case_study.py

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  All evaluations complete. Summaries written to output/."
echo "═══════════════════════════════════════════════════════════════"
