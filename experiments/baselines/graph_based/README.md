# Graph-Based Baselines

Provenance graph 기반 APT 탐지/재구성 시스템.

## 대상 시스템

### OCR-APT
- **원리**: Optimized Causal Relationship for APT — provenance graph 위에서 causal edge를
  최적화해 공격 체인 재구성.
- **입력**: Sysmon 이벤트 → process/file/network 노드 + edge
- **출력**: Subgraph per attack + reconstructed chain
- **Repo**: TODO (원 저자 repo URL 확인 후 적어둘 것)

### MARLIN
- **원리**: Multi-stAge attack Reconstruction via LINeage. 공격 지점에서 역추적하여
  multi-stage chain 복원.
- **입력**: Provenance graph + seed alert
- **출력**: Reconstructed attack subgraph
- **Repo**: TODO

## 통합 방식

1. **변환 스크립트** (`sysmon_to_provenance.py`) — 우리 데이터셋의 Sysmon JSON을
   해당 baseline이 요구하는 provenance graph 포맷으로 변환.
2. **외부 모델 호출** — `subprocess`로 외부 Python/Java 실행, 결과 JSON 파싱.
3. **어댑터** — 결과를 `BaselinePrediction` 형태로 반환.

## 주의

- 외부 repo의 라이선스 확인 (MIT/Apache/BSD 외에는 사용 불가)
- 공정 비교를 위해 **같은 GT + 같은 입력 데이터셋 + 같은 metric** 사용
- 각 baseline의 hyperparameter는 원 논문 권장값 유지 (우리가 튜닝하지 않음)

## TODO

- [ ] OCR-APT 원 repo 클론 + 라이선스 확인
- [ ] MARLIN 원 repo 클론 + 라이선스 확인
- [ ] `sysmon_to_provenance.py` 구현
- [ ] 각 baseline별 어댑터 (`ocr_apt_adapter.py`, `marlin_adapter.py`)
