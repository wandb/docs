---
title: Anonymous mode
description: W&B アカウントなしでデータを ログ および可視化する
menu:
  default:
    identifier: ja-guides-models-app-settings-page-anon
    parent: settings
weight: 80
---

誰でも簡単に実行できるようにしたいコードを公開していますか？ 匿名モードを使用すると、W&B のアカウントを最初に作成しなくても、誰でもあなたのコードを実行し、W&B のダッシュボードを確認し、結果を可視化できます。

匿名モードで結果を記録できるようにするには、以下のようにします。

```python
import wandb

wandb.init(anonymous="allow")
```

たとえば、次のコードスニペットは、W&B で Artifacts を作成およびログに記録する方法を示しています。

```python
import wandb

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

[ノートブックの例](https://colab.research.google.com/drive/1nQ3n8GD6pO-ySdLlQXgbz4wA3yXoSI7i)を試して、匿名モードの動作を確認してください。
