---
title: 複数のメトリクスの最適化
menu:
  support:
    identifier: ja-support-kb-articles-optimizing_multiple_metrics
support:
- sweeps
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

1 つの run で複数のメトリクスを最適化するには、各メトリクスの重み付き和を使います。

```python
with wandb.init() as run:
  # 個々のメトリクスをログする
  metric_a = run.summary.get("metric_a", 0.5)
  metric_b = run.summary.get("metric_b", 0.7)
  # 必要に応じて他のメトリクスもログする
  metric_n = run.summary.get("metric_n", 0.9)

  # メトリクスを重み付きで組み合わせる
  # 最適化の目標に合わせて重みを調整する
  # 例えば、metric_a と metric_n をより重視したい場合:
  metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
  run.log({"metric_combined": metric_combined})
```

新しい合成メトリクスをログし、最適化の目標として設定します:

```yaml
metric:
  name: metric_combined
  goal: minimize
```