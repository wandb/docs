---
title: sweep 内の run からモデルを最適にログするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-best_log_models_runs_sweep
support:
- アーティファクト
- スイープ
toc_hide: true
type: docs
url: /support/:filename
---

[スイープ]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) でモデルをログする効果的な方法のひとつは、スイープ用のモデル artifact を作成することです。各バージョンはスイープからの異なる run を表します。以下のように実装します。

```python
wandb.Artifact(name="sweep_name", type="model")
```