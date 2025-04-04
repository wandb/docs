---
title: What is the difference between `.log()` and `.summary`?
menu:
  support:
    identifier: ko-support-kb-articles-difference_log_summary
support:
- Charts
toc_hide: true
type: docs
url: /support/:filename
---

요약은 테이블에 표시되고, 로그는 향후 플롯을 위해 모든 값을 저장합니다.

예를 들어 정확도가 변경될 때마다 `wandb.log`를 호출합니다. 기본적으로 `wandb.log()`는 해당 메트릭에 대해 수동으로 설정하지 않는 한 요약 값을 업데이트합니다.

산점도 및 평행 좌표 플롯은 요약 값을 사용하는 반면, 선 플롯은 `.log`로 기록된 모든 값을 표시합니다.

일부 사용자는 기록된 가장 최근의 정확도 대신 최적의 정확도를 반영하도록 요약을 수동으로 설정하는 것을 선호합니다.
