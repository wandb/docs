---
title: How do I best log models from runs in a sweep?
menu:
  support:
    identifier: ja-support-kb-articles-best_log_models_runs_sweep
support:
- artifacts
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

[sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) でモデルをログ記録する効果的なアプローチの1つは、sweep のモデル Artifact を作成することです。各 バージョン は、sweep の異なる run を表します。次のように実装します。

```python
wandb.Artifact(name="sweep_name", type="model")
```
