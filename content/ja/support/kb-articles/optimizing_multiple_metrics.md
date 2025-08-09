---
title: 複数のメトリクスの最適化
menu:
  support:
    identifier: ja-support-kb-articles-optimizing_multiple_metrics
support:
- スイープ
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

複数のメトリクスを 1 つの run で最適化するには、各メトリクスの重み付き合計を使います。

```python
with wandb.init() as run:
  # 個々のメトリクスをログする
  metric_a = run.summary.get("metric_a", 0.5)
  metric_b = run.summary.get("metric_b", 0.7)
  # ... 必要に応じて他のメトリクスもログする
  metric_n = run.summary.get("metric_n", 0.9)

  # メトリクスに重みをかけて合成する
  # 最適化の目的に合わせて重みを調整する
  # 例えば metric_a と metric_n を重視したい場合:
  metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
  run.log({"metric_combined": metric_combined})
```

新しい合成メトリクスをログし、それを最適化の目的として設定します:

```yaml
metric:
  name: metric_combined
  goal: minimize
```