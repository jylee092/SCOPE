# Baseline Comparison — 3차 실험

GLIDE를 4가지 baseline 계열과 비교. 각 baseline은 별도 구현/외부 repo 통합이 필요하므로
**어댑터 패턴**으로 구조를 통일.

## 4가지 Baseline

| 폴더 | 계열 | 대표 모델 | 비교 포인트 |
|---|---|---|---|
| `event_level/` | Event-level 분류 | Sysmon 이벤트 단위 SentenceTransformer + FAISS | 그룹 추상화의 가치 |
| `graph_based/` | Provenance graph | **OCR-APT**, **MARLIN** | 엄격한 그래프 vs 유연한 그룹 |
| `llm_based/` | LLM 직접 추론 | **SHIELD** | 구조적 추론 없는 순수 LLM |
| `ttp_sequence/` | TTP sequence model | 사전학습된 시퀀스 모델 | 학습 기반 vs 호환성 기반 |

## 통일된 인터페이스

각 baseline은 `common/adapter.py`의 `BaselineAdapter` 클래스를 상속하여 구현:

```python
class BaselineAdapter:
    def predict(self, scenario_json_path: Path) -> BaselinePrediction:
        """시나리오 하나를 입력받아 예측 결과 반환."""
        ...

class BaselinePrediction:
    tactic_sequence: list[str]        # 예측된 tactic 시퀀스
    technique_sequence: list[str]     # 예측된 technique 시퀀스 (가능한 경우)
    per_group_topk: list[list[dict]]  # 그룹별 Top-K (GLIDE 비교용, 가능한 경우)
```

이렇게 통일하면 `common/metrics.py`가 모든 baseline을 같은 방식으로 평가.

## 출력 경로

```
Final_Code/output/baselines/
├── event_level/atomic/.../result.json
├── graph_based/ocr_apt/atomic/.../result.json
├── graph_based/marlin/atomic/.../result.json
├── llm_based/shield/atomic/.../result.json
└── ttp_sequence/atomic/.../result.json
```

## 평가

`full` variant의 annotation을 **공용 GT**로 사용.

```bash
python experiments/baselines/run_all.py
python experiments/baselines/aggregate.py
```

`aggregate.py`는 GLIDE(full) vs 각 baseline의 tactic F1, chain Jaccard, Hit@K(가능한 경우)를
나란히 비교 테이블로 출력.

## 구현 전략

| Baseline | 구현 방식 |
|---|---|
| `event_level` | **완전 구현** — 기존 SentenceTransformer + FAISS 재사용, 이벤트 단위로 쿼리 |
| `graph_based` | **외부 repo 클론** 후 어댑터로 I/O 연결 (provenance graph 생성 스크립트 포함) |
| `llm_based` | **프롬프트 직접 작성** — 원시 로그를 Gemini에 주고 attack chain 요약 요구 |
| `ttp_sequence` | **pre-trained 모델 활용** 또는 간단한 HMM/LSTM 구현 |

각 폴더의 `README.md`에 세부 계획.
