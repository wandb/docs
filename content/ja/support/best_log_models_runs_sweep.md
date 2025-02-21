---
title: How do I best log models from runs in a sweep?
menu:
  support:
    identifier: ja-support-best_log_models_runs_sweep
tags:
- artifacts
- sweeps
toc_hide: true
type: docs
---

[sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) でモデルをログ記録する効果的なアプローチの 1 つは、sweep 用のモデル Artifact を作成することです。各バージョンは、sweep の異なる run を表します。次のように実装します。

```python
wandb.Artifact(name="sweep_name", type="model")
```
