---
title: Optimizing multiple metrics
menu:
  support:
    identifier: ko-support-optimizing_multiple_metrics
tags:
- sweeps
- metrics
toc_hide: true
type: docs
---

단일 run 에서 여러 메트릭을 최적화하려면 개별 메트릭의 가중치 합계를 사용하세요.

```python
metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
wandb.log({"metric_combined": metric_combined})
```

새로운 결합된 메트릭을 로그하고 최적화 목표로 설정하세요.

```yaml
metric:
  name: metric_combined
  goal: minimize
```
