---
title: Sweep 内の Runs から Models をログする最善の方法は何ですか？
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

[sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) で モデル を ログ する有効な方法のひとつは、その sweep 用に モデル アーティファクト を作成することです。各 バージョン は、その sweep の異なる run を表します。次のように実装します:

```python
wandb.Artifact(name="sweep_name", type="model")
```