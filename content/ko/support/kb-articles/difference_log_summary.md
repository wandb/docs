---
title: .log() 과 .summary 의 차이점은 무엇인가요?
menu:
  support:
    identifier: ko-support-kb-articles-difference_log_summary
support:
- 차트
toc_hide: true
type: docs
url: /support/:filename
---

요약은 표에 표시되며, 로그는 모든 값을 저장해 추후 플로팅에 사용할 수 있습니다.

예를 들어, 정확도가 변경될 때마다 `run.log()`를 호출하세요. 기본적으로 `run.log()`는 해당 metric의 값을 직접 설정하지 않는 한 summary 값도 함께 업데이트합니다.

산점도와 병렬 좌표 플롯은 summary 값을 사용하고, 선형 플롯은 `run.log`로 기록된 모든 값을 보여줍니다.

일부 사용자는 최근 로그된 정확도 대신 최적의 정확도를 반영하기 위해 summary를 직접 설정하는 것을 선호합니다.