---
title: run の sweep からモデルを最適にログする方法は？
menu:
  support:
    identifier: ja-support-kb-articles-best_log_models_runs_sweep
support:
  - artifacts
  - sweeps
toc_hide: true
type: docs
url: /ja/support/:filename
---
モデルを [sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) でログする効果的なアプローチの一つとして、sweep 用のモデル アーティファクトを作成する方法があります。各バージョンは sweep からの異なる run を表します。次のように実装します：

```python
wandb.Artifact(name="sweep_name", type="model")
```