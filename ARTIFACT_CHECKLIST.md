# Artifact Push Checklist

리포지토리에 push 하기 전 마지막 확인용 체크리스트.

---

## 1. 자동으로 정리된 항목 (.gitignore 처리됨)

다음은 **로컬 디스크에는 남아있지만 git push 시 제외**됩니다:

- `__pycache__/`, `.pytest_cache/` (Python 컴파일/테스트 캐시)
- `Dataset/` (799 MB OTRF) → `scripts/setup_data.sh` 가 fetch
- `_sigma_rules/` (44 MB) → 위 스크립트가 fetch
- `output/_snapshots/`, `output/_pre_*/`, `output/_v2_snapshot/` (개발 백업)
- `output/_robustness/` (1.7 GB drop 변형 결과들 — 재실행 가능)
- `output/atomic/**/*_Finale_dataset.csv`, `output/compound/**/*_Finale_dataset.csv` (각 ~30 MB, 재생성 가능한 정규화 CSV)
- `run_log*.txt` (개발 로그)
- `output (2).zip`, `output.zip`, 기타 `*.zip`
- `_skipped/`, `_tmp_*.py`
- `output/v16_*`, `output/v17_*`, `output/v18_*`, `output/v19_*`, `output/v21_*` 등 paper 에 인용 안 되는 sweep
- `.env`, `*.key`, `secrets.json`

## 2. 수동 처리된 항목

- ✅ `config.py` 의 `_LOCAL_API_KEY` 제거 (Gemini 키 하드코딩 삭제)
- ✅ 환경변수 `GEMINI_API_KEY` 만 사용하도록 변경
- ✅ 캐시가 있으면 API 키 없이도 reproduction 가능하도록 안내 문구 추가
- ✅ Author 이름 / 이메일 / 기관 grep — 0 건 (anonymous OK)
- ✅ 개인 경로 (C:/Users/..., D:/Lab/...) grep — 소스 코드에는 0 건

## 3. push 할 실제 콘텐츠 (예상 ~80 MB)

```
Final_Code/
├── README.md                   ★ artifact 가이드
├── LICENSE                     ★ MIT
├── ARTIFACT_CHECKLIST.md       (이 파일 — push 후 삭제 가능)
├── requirements.txt            ★ Python 의존성
├── .gitignore                  ★
├── config.py                   ★ (API key 제거됨)
├── main.py                     ★ end-to-end driver
├── pipeline/                   ★ core SCOPE 모듈
├── experiments/                ★ 평가 + baseline
├── scripts/
│   ├── setup_data.sh           ★ 공개 데이터셋 fetch
│   └── run_all.sh              ★ 일괄 reproduction
├── Technique Rule/             ★ anchor rules
├── TTP_Data/                   ★ MITRE CSV + campaign library
└── output/                     ★ 사전 계산 결과
    ├── _cache/                 LLM description cache (32 MB)
    ├── _ablation/              ablation 결과 JSONs
    ├── _strict_metrics.json    App J 표
    ├── _q5_scope_timings.json  Q5 timing
    ├── _robustness_scores.json Q2 결과
    ├── _novelty_scores.json    Q3 결과
    ├── eval_v2_results.json    Table 2 backing
    ├── eval_*.csv              집계 CSV
    ├── v20_XZ_sweep_results.json
    ├── v22_alpha_bypass_sweep_results.json
    ├── viterbi_tune_sweep_results.json
    ├── atomic/                 35 시나리오 (CSV 제외)
    ├── compound/               2 compound 시나리오
    └── baselines/              4 baseline 결과
```

## 4. push 수동 단계 (사용자가 진행)

```bash
cd "D:/Lab/EDR_Agent/Paper/CCS_Project-main/Final_Code"

# 첫 push 라면:
git init
git add .
git status                    # 의도치 않은 파일이 staged 되었는지 확인
git commit -m "Initial artifact for CCS 2026 SCOPE submission"
git remote add origin <your-github-url>
git branch -M main
git push -u origin main

# 이미 repo 가 있다면:
git add .
git status
git commit -m "Artifact: code, cache, pre-computed eval outputs"
git push
```

push 후 anonymous.4open.science 에 등록하시면 anonymous mirror 가 생성됩니다.

## 5. push 직전 마지막 확인 한 줄

```bash
git status                    # untracked files / staged size
du -sh $(git ls-files | head)  # tracked file 크기 sanity check
```

특히 `git status` 에 다음 항목이 **나타나지 않아야** 합니다:
- `.env`, `*.key`
- `Dataset/`, `_sigma_rules/`
- `output/_snapshots/`, `output/_robustness/`
- `*_Finale_dataset.csv`
- `run_log_*.txt`
- API 키 (`AIzaSy...`)
