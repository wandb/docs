---
title: sweep 内の run からモデルを最適にログする方法は？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
- スイープ
---

[sweep]({{< relref "/guides/models/sweeps/" >}}) でモデルをログする効果的な方法のひとつは、スイープ用の model アーティファクトを作成することです。それぞれのバージョンは、スイープの異なる run を表します。以下のように実装します。

```python
wandb.Artifact(name="sweep_name", type="model")
```