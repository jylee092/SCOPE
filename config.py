"""
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_FOLDER = BASE_DIR / "Dataset"
RULE_FOLDER = BASE_DIR / "Technique Rule"
MITRE_CSV_PATH = BASE_DIR / "TTP_Data" / "Final_merged_mitre_attack_data.csv"
OUTPUT_BASE_DIR = BASE_DIR / "output"

DATASET_NAME: str = ""
DATASET_FILE: Path = None
OUTPUT_DIR: Path = None
FINALE_CSV_PATH: Path = None
FEATURE_RESULT_JSON_PATH: Path = None
TTP_MAPPING_JSON_PATH: Path = None
ANNOTATION_JSON_PATH: Path = None
VITERBI_JSON_PATH: Path = None
EVAL_JSON_PATH: Path = None


def configure_dataset(dataset_path: Path) -> None:
    """Reconfigure all derived paths for a single scenario JSON.

    Mirrors the relative path under DATASET_FOLDER inside OUTPUT_BASE_DIR
    so each scenario's intermediate and final outputs live in a parallel
    output/<rel>/<stem>/ directory."""
    global DATASET_NAME, DATASET_FILE, OUTPUT_DIR
    global FINALE_CSV_PATH, FEATURE_RESULT_JSON_PATH, TTP_MAPPING_JSON_PATH
    global ANNOTATION_JSON_PATH, VITERBI_JSON_PATH, EVAL_JSON_PATH

    dataset_path = Path(dataset_path).resolve()
    rel = dataset_path.relative_to(
        DATASET_FOLDER).with_suffix("")  # e.g., campaignA/foo
    stem = dataset_path.stem

    DATASET_NAME = stem
    DATASET_FILE = dataset_path
    OUTPUT_DIR = OUTPUT_BASE_DIR / rel
    FINALE_CSV_PATH = OUTPUT_DIR / f"{stem}_Finale_dataset.csv"
    FEATURE_RESULT_JSON_PATH = OUTPUT_DIR / f"{stem}_feature_result.json"
    TTP_MAPPING_JSON_PATH = OUTPUT_DIR / f"{stem}_ttp_mapping.json"
    ANNOTATION_JSON_PATH = OUTPUT_DIR / f"{stem}_annotation.json"
    VITERBI_JSON_PATH = OUTPUT_DIR / f"{stem}_viterbi.json"
    EVAL_JSON_PATH = OUTPUT_DIR / f"{stem}_eval.json"


# Set GEMINI_API_KEY in your shell environment before running:
#   export GEMINI_API_KEY=...                 # bash/zsh
#   $env:GEMINI_API_KEY = "..."               # PowerShell
# All description-generation calls are cached on disk, so reproducing
# the reported numbers does not require an API key -- see README.md.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("[info] GEMINI_API_KEY is not set; pipeline will run from the on-disk LLM cache only.")
EMBED_MODEL_NAME = "all-mpnet-base-v2"
GEMINI_MODEL = "models/gemini-2.5-flash"

TOP_K = 5

GROUPING_BEFORE_SEC = 10
GROUPING_AFTER_SEC = 30
GROUPING_HOP_UP = 2
GROUPING_HOP_DOWN = 3
GROUPING_APPLY_FILTER = True
GROUPING_USE_SHARED_ENTITY = True
GROUPING_MAX_ANCHORS_PER_RULE = 20
MERGE_OVERLAP_THRESHOLD = 0.5
DROP_FILTER_FAILED_GROUPS = True

VITERBI_MIN_SIM_GATE = 0.0
VITERBI_MAX_GROUPS_AFTER_GATE = 0

VITERBI_TRANSITION_WEIGHT = 0.5
TACTIC_ANOMALY_THRESHOLD = 0.1
VITERBI_BEAM_K = 5
VITERBI_MAX_SKIP = 0
VITERBI_SKIP_PENALTY = 0.25
SELF_LOOP_TID_PENALTY = 1.0

# v14 margin-conditional α:
VITERBI_MARGIN_GATED_ALPHA   = False

VITERBI_MARGIN_LOW           = 0.003
VITERBI_MARGIN_HIGH          = 0.015
VITERBI_ALPHA_LOW_MARGIN     = 0.25
VITERBI_ALPHA_HIGH_MARGIN    = 0.03

VITERBI_SIM_GATED_ALPHA      = False

VITERBI_SIM_MARGIN_LOW       = 0.03

VITERBI_SIM_MARGIN_HIGH      = 0.10

VITERBI_ALPHA_LOW_SIM        = 0.5

VITERBI_ALPHA_HIGH_SIM       = 0.1


# P_sem removed (CCS->C&S revision): bi-encoder cosine is symmetric and its
# contribution is exactly zero on this corpus (0.6833 with vs without). The
# transition model is now two principled, directional axes: tactical (P_tac)
# and causal (P_cau). See experiments/ccs_revision/_nopsem_check.py.
USE_SEMANTIC_SCORING = False
USE_CAUSAL_SCORING = True
SEMANTIC_BACKEND     = "bi-encoder"
SEMANTIC_MODEL       = "basel/ATTACK-BERT"
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#   s = σ(β · (cos − c0))
SEM_CALIBRATION        = "linear"

SEM_SIGMOID_CENTER     = 0.5
SEM_SIGMOID_SCALE      = 8.0
W_TAC = 0.5
W_SEM = 0.0   # P_sem removed (zero contribution; symmetric). Renormalizes to tac/cau.
W_CAU = 0.2
USE_CE_EMISSION_RERANK = False
CE_RERANK_WIDTH = 20
CE_RERANK_WEIGHT = 0.0
BM25_WEIGHT = 0.15
BM25_RERANK_WIDTH = 30
SIGNATURE_WEIGHT = 0.8
SIGNATURE_RERANK_WIDTH = 10
FAMILY_BOOST         = 0.15
FAMILY_BOOST_WIDTH   = 10
VITERBI_HARD_TACTIC_FILTER = True

# v20: Emission-confidence bypass (X+Z hybrid).
EMISSION_BYPASS_SIM_THRESHOLD = 0.75
RULE_TID_PRIOR = 1.15
RULE_TACTIC_PRIOR = 1.05

CAMPAIGN_FOLDER = BASE_DIR / "TTP_Data" / "Campaign"
CACHE_DIR = BASE_DIR / "output" / "_cache"

SAMPLE_PER_TECHNIQUE = 50

MAX_GROUPS_PER_SCENARIO = 80
