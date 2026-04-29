"""
전역 설정 — 경로, 모델명, 하이퍼파라미터.
환경변수 GEMINI_API_KEY로 API 키 주입.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_FOLDER = BASE_DIR / "Dataset"
RULE_FOLDER = BASE_DIR / "Technique Rule"
MITRE_CSV_PATH = BASE_DIR / "TTP_Data" / "Final_merged_mitre_attack_data.csv"
OUTPUT_BASE_DIR = BASE_DIR / "output"

# 현재 처리 중인 데이터셋 경로 (configure_dataset()이 채움)
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
    """데이터셋 .json 파일 경로를 받아 관련 경로를 전부 재계산.
    DATASET_FOLDER 기준 상대 경로를 OUTPUT_BASE_DIR 아래에 동일 구조로 반영."""
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
# the reported numbers does not require an API key — see README.md.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("[info] GEMINI_API_KEY is not set; pipeline will run from the on-disk LLM cache only.")
# v15 실험: ATTACK-BERT 는 Top-5 0.95 → 0.73 으로 대폭 악화 (CTI-report 분포에 편향,
# Sysmon 기반 긴 description 과 부정합). mpnet 복귀. 2026-04-22.
EMBED_MODEL_NAME = "all-mpnet-base-v2"
GEMINI_MODEL = "models/gemini-2.5-flash"  # v11: 2.0 quota 소진 + 프롬프트 대폭 변경
TOP_K = 5

GROUPING_BEFORE_SEC = 10
GROUPING_AFTER_SEC = 30
GROUPING_HOP_UP = 2
GROUPING_HOP_DOWN = 3
GROUPING_APPLY_FILTER = True
GROUPING_USE_SHARED_ENTITY = True
# v2 설정: anchor cap 20, group cap 80, skip 0.25, self-loop 0.3, rule prior 1.25
# 이 조합이 chain 구조 지표(tac_lcs 0.66, step_cov 0.55)에서 가장 강함.
GROUPING_MAX_ANCHORS_PER_RULE = 20
MERGE_OVERLAP_THRESHOLD = 0.5
DROP_FILTER_FAILED_GROUPS = True

# v7: weight sweep 결과로 확정.
# transition_weight 0.5→0.1 (emission 지배), self_loop 0.3→1.0 (동일 TID 허용), max_skip=0.
# 이 조합에서 Viterbi가 FAISS top-1 대비 per-group plausibility +0.029 rerank 이득.
# v13 실험: sim>=0.6 gate 시도 → 실제 TP 그룹(sim 0.55~0.60)을 29% drop하여
# Viterbi pick plausibility -14%p, chain metrics 악화. 롤백.
VITERBI_MIN_SIM_GATE = 0.0
VITERBI_MAX_GROUPS_AFTER_GATE = 0

VITERBI_TRANSITION_WEIGHT = 0.5
TACTIC_ANOMALY_THRESHOLD = 0.1
VITERBI_BEAM_K = 5
VITERBI_MAX_SKIP = 0
VITERBI_SKIP_PENALTY = 0.25
SELF_LOOP_TID_PENALTY = 1.0

# v14 margin-conditional α:
# emission vs transition 가중치를 그룹별 confidence_margin 에 맞춰 동적으로 조정.
# margin(= p_ttp(top-1) - p_ttp(top-2)) 분포는 0.000~0.028 에 집중 (softmax 평탄).
# 실측 분포: p25≈0.0017, p50≈0.0038, p75≈0.0084, p90≈0.0158
#   margin < MARGIN_LOW   → α = ALPHA_LOW_MARGIN  (top-1이 애매 → transition 에 위임)
#   margin > MARGIN_HIGH  → α = ALPHA_HIGH_MARGIN (top-1 확실 → FAISS 고수)
#   중간                   → VITERBI_TRANSITION_WEIGHT 그대로
# regress (FAISS hit → Viterbi miss) 방지 + unrecovered (top-5 에 있는데 못 꺼냄) 회수 목적.
VITERBI_MARGIN_GATED_ALPHA   = False   # v14 실험: 어떤 임계값 조합도 fixed α=0.1 대비 악화 → 비활성 (2026-04-22)
VITERBI_MARGIN_LOW           = 0.003
VITERBI_MARGIN_HIGH          = 0.015
VITERBI_ALPHA_LOW_MARGIN     = 0.25
VITERBI_ALPHA_HIGH_MARGIN    = 0.03

# v16: similarity 원값 기반 margin gating.
# p_ttp softmax margin 은 분포가 [0.0017, 0.015] 로 평탄해 gate 판별력 없음.
# raw similarity top1-top2 차이는 [0.01, 0.15] 로 훨씬 뾰족함 → 실제 uncertainty
# 신호로 동작. sim_margin 작음 = top-1 애매함 → transition 에 위임 (α↑).
VITERBI_SIM_GATED_ALPHA      = False  # v16b sweep 에서 -1.4pp 확인 → OFF
VITERBI_SIM_MARGIN_LOW       = 0.03   # 이 이하 면 ambiguous
VITERBI_SIM_MARGIN_HIGH      = 0.10   # 이 이상 이면 confident
VITERBI_ALPHA_LOW_SIM        = 0.5    # ambiguous → transition 더 신뢰
VITERBI_ALPHA_HIGH_SIM       = 0.1    # confident → emission 고수

USE_SEMANTIC_SCORING = True
USE_CAUSAL_SCORING = True
# Semantic transition 백엔드. 'cross-encoder' or 'bi-encoder'.
# v14 실험: cross-encoder/ms-marco 는 보안 도메인 부적합 → bi-encoder (basel/ATTACK-BERT)
# 로 교체. SMET 저자의 attack-description→ATT&CK 매핑 모델이라 task 형식 일치.
SEMANTIC_BACKEND     = "bi-encoder"
SEMANTIC_MODEL       = "basel/ATTACK-BERT"
# 이전 cross-encoder 설정 (비교/롤백용).
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# v16: P_sem 재교정. 기존 linear [-1,1]→[0,1] 은 ATTACK-BERT 코사인 분포가
# [0.3, 0.7] 좁게 몰려 있어 매핑 결과 [0.65, 0.85] 평탄화 → geometric mean 에서
# 판별력 없음. sigmoid 로 교체해 중심 주변 gradient 를 키운다.
#   s = σ(β · (cos − c0))
# True 면 sigmoid, False 면 기존 linear 유지 (롤백용).
SEM_CALIBRATION        = "linear"    # v16b: sigmoid 효과 0 → linear 기본
SEM_SIGMOID_CENTER     = 0.5
SEM_SIGMOID_SCALE      = 8.0
W_TAC = 0.5
W_SEM = 0.3
W_CAU = 0.2
# v7 실험: Cross-encoder emission rerank로 실험했으나 MS-MARCO 도메인 기반 CE가
# 보안 description 매칭에 부적합해 Plausible H@1이 0.56→0.37로 대폭 하락. 비활성.
USE_CE_EMISSION_RERANK = False
CE_RERANK_WIDTH = 20
CE_RERANK_WEIGHT = 0.0
# BM25 + dense hybrid — keyword match. v11에서 0.3은 top-1 편향 강함 → 0.15로 완화.
BM25_WEIGHT = 0.15
BM25_RERANK_WIDTH = 30
# v15: TID-specific artifact signature (Technique Rule 에서 추출한 contains_any 키워드)
# 로 상위 후보 재정렬. FAISS+BM25 가 조밀한 후보들 사이에서 artifact 증거로 tie-break.
SIGNATURE_WEIGHT = 0.8
SIGNATURE_RERANK_WIDTH = 10
# v19 A1: family-consensus boost. Top-N 안에서 같은 parent family (T1003.*) 공유
# 후보 수만큼 곱해주는 multiplicative boost. 0 = 비활성.
FAMILY_BOOST         = 0.15
FAMILY_BOOST_WIDTH   = 10
# v19 A2: Hard tactic-mismatch filter. R9 (forbidden_src/forbidden_pairs) 전이를
# soft weight 0.02 대신 Viterbi beam 에서 완전 제외.
VITERBI_HARD_TACTIC_FILTER = True

# v20: Emission-confidence bypass (X+Z hybrid).
# sim(top-1) >= threshold 인 그룹은 FAISS top-1 그대로 사용 (emission 신뢰).
# sim(top-1) < threshold 인 그룹은 Viterbi 선택 유지 (emission 불확실 → 구조 활용).
# None 이면 비활성. 0.75 = top-1 sim p75 (상위 ~25% confident 그룹만 bypass).
EMISSION_BYPASS_SIM_THRESHOLD = 0.75
# Rule family prior — rule이 지정한 TID의 parent와 같은 family 후보 soft boost
RULE_TID_PRIOR = 1.15
RULE_TACTIC_PRIOR = 1.05

CAMPAIGN_FOLDER = BASE_DIR / "TTP_Data" / "Campaign"
CACHE_DIR = BASE_DIR / "output" / "_cache"

# 한 시나리오 내에서 동일 rule-technique의 그룹을 몇 개까지 LLM 분석할지.
# compound 시나리오에서 진성 TP를 자르지 않도록 기본값을 높게 잡음. 0 = 무제한.
SAMPLE_PER_TECHNIQUE = 50

# v2: 80 (compound에서 TP 잘림 있지만 structure가 가장 좋음).
MAX_GROUPS_PER_SCENARIO = 80
