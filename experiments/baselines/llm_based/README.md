# LLM-Based Baseline — SHIELD

LLM에게 **원시 로그를 직접** 주고 공격 체인을 요약하도록 요구. 구조적 추론(그룹/Viterbi)
없이 LLM의 semantic reasoning만으로 얼마나 재구성 가능한지 측정.

## 접근

1. 시나리오의 Sysmon 이벤트를 시간순으로 직렬화하여 LLM 컨텍스트로 투입
2. 프롬프트로 "MITRE ATT&CK tactic 시퀀스와 involved techniques 추출" 요구
3. LLM 응답을 파싱하여 tactic/technique 시퀀스 추출

## 주의

- **컨텍스트 윈도우 한계**: 대형 시나리오(apt29 100K+ 이벤트)는 잘라서 처리하거나
  중요 이벤트만 필터링
- **프롬프트 공정성**: GLIDE가 쓰는 Gemini와 동일 모델(`gemini-2.0-flash`) 사용
- **Grounding 부족**: LLM이 halluclnate할 수 있음 — 이 점이 GLIDE 대비 약점이 드러나는 지점

## SHIELD 원 논문

TODO: DOI/arXiv 링크 및 프롬프트 템플릿 발췌

## 어댑터

`adapter.py` 참고.
