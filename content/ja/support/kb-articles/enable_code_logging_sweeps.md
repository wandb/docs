---
menu:
  support:
    identifier: ja-support-kb-articles-enable_code_logging_sweeps
support:
- sweeps
title: How do I enable code logging with Sweeps?
toc_hide: true
type: docs
url: /support/:filename
---

To enable code logging for sweeps, add `wandb.log_code()` after initializing the W&B Run. This action is necessary even when code logging is enabled in the W&B profile settings. For advanced code logging, refer to the [docs for `wandb.log_code()` here]({{< relref path="/ref/python/sdk/classes/run#log_code" lang="ja" >}}).