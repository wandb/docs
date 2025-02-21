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

モデルを [sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) にログする効果的なアプローチの一つは、スイープのためのモデルアーティファクトを作成することです。それぞれのバージョンは、スイープの異なる run を表します。以下のように実装します:

```python
wandb.Artifact(name="sweep_name", type="model")
```