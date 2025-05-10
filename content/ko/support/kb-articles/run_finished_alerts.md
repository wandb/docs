---
title: Do "Run Finished" alerts work in notebooks?
menu:
  support:
    identifier: ko-support-kb-articles-run_finished_alerts
support:
- alerts
- notebooks
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: run_finished_alerts
---
아니요. **Run Finished** 알림(사용자 설정에서 **Run Finished** 설정으로 활성화됨)은 Python 스크립트에서만 작동하며 각 셀 실행에 대한 알림을 피하기 위해 Jupyter 노트북 환경에서는 꺼진 상태로 유지됩니다.

노트북 환경에서는 대신 `wandb.alert()`를 사용하세요.
