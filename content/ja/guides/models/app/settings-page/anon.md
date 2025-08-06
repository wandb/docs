---
title: 匿名モード
description: W&B アカウントなしでデータをログし可視化する
menu:
  default:
    identifier: anon
    parent: settings
weight: 80
---

公開したいコードを誰でも簡単に実行できるようにしたい場合は、anonymous モードを利用しましょう。anonymous モードを使えば、誰でも W&B アカウントを作成せずにあなたのコードを実行し、W&B ダッシュボードを見て、結果を可視化できます。

anonymous モードで結果をログに記録するには、次のようにします：

```python
import wandb

wandb.init(anonymous="allow")
```

例えば、下記のコードスニペットは、W&B で Artifact を作成しログに記録する方法を示しています：

```python
import wandb

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

[サンプルノートブックを試して](https://colab.research.google.com/drive/1nQ3n8GD6pO-ySdLlQXgbz4wA3yXoSI7i)、anonymous モードがどのように動作するか確認してみましょう。