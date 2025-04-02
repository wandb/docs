---
title: How do I enable code logging with Sweeps?
menu:
  support:
    identifier: ko-support-kb-articles-enable_code_logging_sweeps
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

스윕에 대한 코드 로깅을 활성화하려면 W&B Run을 초기화한 후 `wandb.log_code()`를 추가하세요. 이 작업은 W&B 프로필 설정에서 코드 로깅이 활성화된 경우에도 필요합니다. 고급 코드 로깅에 대해서는 [`wandb.log_code()`에 대한 문서]({{< relref path="/ref/python/run.md#log_code" lang="ko" >}})를 참조하세요.
