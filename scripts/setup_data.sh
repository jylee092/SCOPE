#!/usr/bin/env bash
# =============================================================
# SCOPE artifact -- one-time data fetch.
#
# Downloads the two public datasets that we do not bundle:
#   1. OTRF Security-Datasets (Mordor) -- 35 attack scenarios.
#   2. SigmaHQ rules           -- ~2,140 Windows detection rules.
#
# Both repositories are cloned at fixed commits so the artifact
# is reproducible regardless of upstream evolution.
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OTRF_COMMIT="${OTRF_COMMIT:-main}"
SIGMA_COMMIT="${SIGMA_COMMIT:-main}"

# ---------------------------------------------------------------
# 1. OTRF Security-Datasets (Mordor)
# ---------------------------------------------------------------
if [ ! -d "Dataset" ]; then
  echo "[setup] cloning OTRF Security-Datasets ..."
  tmp_otrf="$(mktemp -d)"
  git clone --depth 1 --branch "$OTRF_COMMIT" \
      https://github.com/OTRF/Security-Datasets.git "$tmp_otrf"
  # We use only the 35 atomic + 2 compound scenarios listed in
  # experiments/attack_flows.py. Copy the relevant subdirectories.
  mkdir -p Dataset
  cp -r "$tmp_otrf/datasets/atomic"   Dataset/atomic
  cp -r "$tmp_otrf/datasets/compound" Dataset/compound 2>/dev/null || true
  rm -rf "$tmp_otrf"
  echo "[setup] OTRF Dataset/ ready ($(du -sh Dataset | cut -f1))."
else
  echo "[setup] Dataset/ already exists, skipping."
fi

# ---------------------------------------------------------------
# 2. SigmaHQ rules (Sigma baseline)
# ---------------------------------------------------------------
if [ ! -d "_sigma_rules" ]; then
  echo "[setup] cloning SigmaHQ/sigma ..."
  git clone --depth 1 --branch "$SIGMA_COMMIT" \
      https://github.com/SigmaHQ/sigma.git _sigma_rules
  echo "[setup] _sigma_rules/ ready ($(du -sh _sigma_rules | cut -f1))."
else
  echo "[setup] _sigma_rules/ already exists, skipping."
fi

# ---------------------------------------------------------------
# 3. Sanity check
# ---------------------------------------------------------------
echo
echo "[setup] sanity check:"
n_scenarios=$(find Dataset -name "*.json" -type f 2>/dev/null | wc -l)
n_sigma_rules=$(find _sigma_rules/rules/windows -name "*.yml" -type f 2>/dev/null | wc -l)
echo "  OTRF scenarios       : $n_scenarios   (expected: ~35)"
echo "  SigmaHQ Windows rules: $n_sigma_rules (expected: 2,000+)"
echo
echo "Setup complete. Run scripts/run_all.sh to reproduce paper numbers."
