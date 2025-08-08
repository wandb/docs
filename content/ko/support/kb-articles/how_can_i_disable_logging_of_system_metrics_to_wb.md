---
title: How can I disable logging of system metrics to W&B?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_disable_logging_of_system_metrics_to_wb
support:
- metrics
- runs
toc_hide: true
type: docs
url: /ko/support/:filename
---

[시스템 메트릭]({{< relref path="/ref/system-metrics.md" lang="ko" >}})의 로깅을 비활성화하려면 `_disable_stats`를 `True`로 설정하세요:

```python
wandb.init(settings=wandb.Settings(x_disable_stats=True))
```