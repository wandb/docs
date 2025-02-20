---
title: Anonymous mode
description: データをログし、W&B アカウントなしで視覚化する
menu:
  default:
    identifier: ja-guides-models-app-settings-page-anon
    parent: settings
weight: 80
---

コードを誰でも簡単に実行できるように公開したいですか？ 匿名モードを使用して、誰かがあなたのコードを実行し、W&B ダッシュボードを見て、W&B アカウントをまず作成することなく結果を視覚化できるようにします。

結果を匿名モードでログとして記録するには、次のようにします:

```python
import wandb

wandb.init(anonymous="allow")
```

例えば、次のコードスニペットは W&B を使ってアーティファクトを作成し、ログする方法を示しています:

```python
import wandb

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

[匿名モードの作動を見るための例のノートブックを試す](https://colab.research.google.com/drive/1nQ3n8GD6pO-ySdLlQXgbz4wA3yXoSI7i) ことができます。