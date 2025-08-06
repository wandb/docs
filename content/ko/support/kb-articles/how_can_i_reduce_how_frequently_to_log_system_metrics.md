---
title: 시스템 메트릭을 얼마나 자주 로그할지 변경하려면 어떻게 해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_reduce_how_frequently_to_log_system_metrics
support:
- 메트릭
- run
toc_hide: true
type: docs
url: /support/:filename
---

[시스템 메트릭]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ko" >}})을(를) 로그하는 빈도를 설정하려면 `_stats_sampling_interval` 값을 초 단위(실수형)로 지정하세요. 기본값은 `10.0`입니다.

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```