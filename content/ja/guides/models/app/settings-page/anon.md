---
title: 匿名モード
description: データをログし、W&B アカウントなしで可視化する
menu:
  default:
    identifier: ja-guides-models-app-settings-page-anon
    parent: settings
weight: 80
---

コードを誰でも簡単に実行できるように公開していますか？ 匿名モードを使用して、誰かがあなたのコードを実行し、W&B ダッシュボードを見て、W&B アカウントを作成することなく結果を視覚化できるようにします。

結果が匿名モードでログに記録されるようにするには、次のようにします：

```python
import wandb

wandb.init(anonymous="allow")
```

例えば、次のコードスニペットは、W&B でアーティファクトを作成し、ログに記録する方法を示しています：

```python
import wandb

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

[例のノートブックを試してみて](https://colab.research.google.com/drive/1nQ3n8GD6pO-ySdLlQXgbz4wA3yXoSI7i)、匿名モードがどのように機能するかを確認してください。