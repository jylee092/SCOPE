# TTP Sequence Modeling Baseline

기존 TTP 시퀀스를 학습한 모델로 공격 체인 분류/완성. 학습 기반 방식이 **새로운
조합(novel chain)**에 대해 GLIDE의 compatibility-based 접근 대비 얼마나 약한지 검증.

## 접근 옵션

1. **HMM** — tactic 상태 × technique 관측을 가정, MITRE CTI campaign 라이브러리에서 학습.
2. **LSTM** — 시퀀스 분류기, 공개 campaign 데이터로 pre-train.
3. **기존 TTP seq 모델** — 공개된 pre-trained 모델이 있다면 활용 (TODO).

## 제약

- 학습 데이터가 MITRE 공식 51개 campaign으로 한정 (우리 Campaign library와 동일)
- 우리 평가 시나리오의 "novel chain" 특성상 학습 분포 밖 케이스 다수 → recall 저하 예상
- 이는 논문의 "chain-level novelty coverage" 주장의 실증 근거

## TODO

- [ ] HMM 구현 (`hmm_adapter.py`) — 제일 간단
- [ ] LSTM 대안 (선택)
- [ ] pre-trained 모델 탐색 (BERT-for-TTP 등)
