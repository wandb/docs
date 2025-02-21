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

単一の run で複数の メトリクス を最適化するには、個々の メトリクス の加重和を使用します。

```python
metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
wandb.log({"metric_combined": metric_combined})
```

新しい組み合わせた メトリクス を ログ に記録し、最適化の目的として設定します。

```yaml
metric:
  name: metric_combined
  goal: minimize
```
