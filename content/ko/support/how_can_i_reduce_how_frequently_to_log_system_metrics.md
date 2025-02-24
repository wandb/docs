---
title: How can I change how frequently to log system metrics?
menu:
  support:
    identifier: ko-support-how_can_i_reduce_how_frequently_to_log_system_metrics
tags:
- metrics
- runs
toc_hide: true
type: docs
---

[시스템 메트릭]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ko" >}})을 기록하는 빈도를 설정하려면 `_stats_sampling_interval`을 초 단위의 숫자로 설정하세요(float로 표현). 기본값: `10.0`입니다.

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```
