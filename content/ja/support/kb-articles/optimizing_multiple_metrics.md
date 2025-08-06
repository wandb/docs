---
title: 複数のメトリクスの最適化
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
- メトリクス
---

複数のメトリクスを 1 つの run で最適化するには、個々のメトリクスの加重和を使用します。

```python
with wandb.init() as run:
  # 個々のメトリクスをログする
  metric_a = run.summary.get("metric_a", 0.5)
  metric_b = run.summary.get("metric_b", 0.7)
  # 必要に応じて他のメトリクスもログ
  metric_n = run.summary.get("metric_n", 0.9)

  # メトリクスを重み付きで合成する
  # 最適化の目的に応じて重みを調整してください
  # たとえば、metric_a と metric_n により重要性を持たせたい場合:
  metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
  run.log({"metric_combined": metric_combined})
```

新しい合成メトリクスをログし、最適化目標として設定します:

```yaml
metric:
  name: metric_combined
  goal: minimize
```