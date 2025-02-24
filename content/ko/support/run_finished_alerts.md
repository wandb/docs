---
title: Do "Run Finished" alerts work in notebooks?
menu:
  support:
    identifier: ko-support-run_finished_alerts
tags:
- alerts
- notebooks
toc_hide: true
type: docs
---

아니요. **Run Finished** 알림 (사용자 설정의 **Run Finished** 설정으로 활성화)은 Python 스크립트에서만 작동하며, 각 셀 실행에 대한 알림을 피하기 위해 Jupyter 노트북 환경에서는 꺼진 상태로 유지됩니다.

노트북 환경에서는 대신 `wandb.alert()`를 사용하세요.
