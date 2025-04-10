---
title: How can I change how frequently to log system metrics?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_reduce_how_frequently_to_log_system_metrics
support:
- metrics
- runs
toc_hide: true
type: docs
url: /support/:filename
---

[시스템 메트릭]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ko" >}})을 로깅하는 빈도를 설정하려면 `_stats_sampling_interval`을 초 단위로 나타낸 부동 소수점 숫자로 설정하세요. 기본값: `10.0`.

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```
