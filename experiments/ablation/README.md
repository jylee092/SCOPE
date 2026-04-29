# Ablation Study — 2차 실험

4개 변종으로 구성된 ablation study. 본 코드(`main.py`, `pipeline/*`, `config.py`)는
**건드리지 않고** 독립 실행.

## 변종

| 이름 | 제거 대상 | 구현 방식 |
|---|---|---|
| `full` | 없음 (baseline) | 기존 `main.py`와 동등 |
| `no_grouping` | 룰 기반 그룹핑 | `ablation_helpers.build_solo_groups()` — 1 이벤트 = 1 그룹 |
| `no_llm` | Gemini description | `ablation_helpers.feature_to_text()` 사용 후 FAISS 쿼리 |
| `top1_only` | Top-K beam + hole-bridging | `topk_viterbi(beam_k=1, max_skip=0)` |

## 출력 경로

```
Final_Code/output/
├── atomic/, compound/...              ← full variant (기존 위치)
├── ablation_no_grouping/atomic/...
├── ablation_no_llm/atomic/...
└── ablation_top1_only/atomic/...
```

`full` variant의 `*_annotation.json`을 GT로 **공용**. `no_grouping`은 그룹이 달라
TTP Hit@K 평가 불가 — Tactic Chain metrics만 비교.

## 실행

전체:
```
cd Final_Code
python experiments/ablation/run_all.py
```

일부 variant만:
```
python experiments/ablation/run_all.py --variants no_llm top1_only
```

특정 시나리오만 (prefix 필터):
```
python experiments/ablation/run_all.py --scenarios atomic/collection
```

## 집계

```
python experiments/ablation/aggregate.py
```
`experiments/ablation/comparison.json` + 콘솔 비교 테이블 출력.

## 전제

- `main.py`가 먼저 실행되어 `full`의 annotation 템플릿이 생성되고 **수동 라벨링 완료**된 상태여야 함.
- LLM/FAISS 캐시는 variant 간 자동 공유 (해시 기반). `no_llm`은 Gemini 호출 안 함.
- `no_grouping` 변종은 그룹 수가 많아 LLM 비용이 가장 큼 — `max_groups=500`으로 상한.
