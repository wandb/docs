---
title: Sweeps 에서 코드 로깅을 어떻게 활성화하나요?
menu:
  support:
    identifier: ko-support-kb-articles-enable_code_logging_sweeps
support:
- 스윕
toc_hide: true
type: docs
url: /support/:filename
---

Sweeps 에서 코드 로깅을 활성화하려면 W&B Run 을 초기화한 후에 `wandb.log_code()` 를 추가하세요. 이 동작은 W&B 프로필 설정에서 코드 로깅이 활성화되어 있더라도 필요합니다. 고급 코드 로깅에 대해서는 [여기에서 `wandb.log_code()` 문서]({{< relref path="/ref/python/sdk/classes/run#log_code" lang="ko" >}})를 참고하세요.