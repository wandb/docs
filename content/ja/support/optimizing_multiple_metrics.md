---
title: Optimizing multiple metrics
menu:
  support:
    identifier: ja-support-optimizing_multiple_metrics
tags:
- sweeps
- metrics
toc_hide: true
type: docs
---

複数のメトリクスを1つの run で最適化するには、それぞれのメトリクスの加重和を使用します。

```python
metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
wandb.log({"metric_combined": metric_combined})
```

新しい組み合わせたメトリクスをログに記録し、それを最適化の目的として設定します。

```yaml
metric:
  name: metric_combined
  goal: minimize
```