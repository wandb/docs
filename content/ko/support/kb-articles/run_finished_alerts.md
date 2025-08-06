---
title: 노트북에서 "Run Finished" 알림이 작동하나요?
menu:
  support:
    identifier: ko-support-kb-articles-run_finished_alerts
support:
- 알림
- 노트북
toc_hide: true
type: docs
url: /support/:filename
---

아니요. **Run Finished** 알림(**Run Finished** 설정은 User Settings에서 활성화할 수 있음)은 Python 스크립트에서만 작동하며, Jupyter Notebook 환경에서는 각 셀 실행마다 알림이 생성되는 것을 방지하기 위해 비활성화되어 있습니다.

노트북 환경에서는 대신 `run.alert()`를 사용하세요.