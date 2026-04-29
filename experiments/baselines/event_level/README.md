# Event-Level Baseline

개별 Sysmon 이벤트 단위로 TTP 분류. GLIDE의 "그룹 추상화"가 주는 가치를 검증하는 가장
직접적인 baseline.

## 구현 방식

GLIDE 파이프라인의 `all-mpnet-base-v2` + FAISS 인덱스를 **그대로 재사용**. 차이점:
- **입력 단위**: 그룹 → 개별 이벤트
- **description**: LLM 생성 안 함, 이벤트 필드를 단순 직렬화 (Image + CommandLine + TargetObject 등)
- **시퀀스 구성**: 이벤트 순서대로 Top-1 technique을 이어붙여 tactic sequence 생성

## 어댑터

`adapter.py` 참고.

## 실행

`run_all.py`가 전체 시나리오에 대해 자동 실행.
