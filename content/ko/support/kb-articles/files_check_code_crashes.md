---
title: 내 코드가 오류 날 때 어떤 파일을 확인해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-files_check_code_crashes
support:
- 로그
toc_hide: true
type: docs
url: /support/:filename
---

문제가 발생한 run 에 대해, 코드가 실행되고 있는 디렉토리의 `wandb/run-<date>_<time>-<run-id>/logs` 에 있는 `debug.log` 와 `debug-internal.log` 를 확인하세요.