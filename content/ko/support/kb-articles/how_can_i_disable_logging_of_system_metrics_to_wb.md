---
title: W&B 에 시스템 메트릭 로그를 비활성화할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_disable_logging_of_system_metrics_to_wb
support:
- 메트릭
- run
toc_hide: true
type: docs
url: /support/:filename
---

[시스템 메트릭]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ko" >}}) 로그를 비활성화하려면, `_disable_stats`를 `True`로 설정하세요:

```python
wandb.init(settings=wandb.Settings(x_disable_stats=True))
```