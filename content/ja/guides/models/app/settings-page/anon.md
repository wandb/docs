---
title: 匿名モード
description: W&B アカウントなしで データ をログし可視化する
menu:
  default:
    identifier: ja-guides-models-app-settings-page-anon
    parent: settings
weight: 80
---

誰でも簡単に実行できるコードを公開したい場合は、anonymous モードを使いましょう。anonymous モードを使うことで、誰でも W&B アカウントを作成せずにコードを実行し、W&B ダッシュボードを見たり、結果を可視化したりできます。

anonymous モードで結果をログできるようにするには、以下のようにします。

```python
import wandb

wandb.init(anonymous="allow")
```

たとえば、以下のコードスニペットでは、W&B で Artifact を作成し、ログする方法を示しています。

```python
import wandb

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

[サンプルノートブックを試してみる](https://colab.research.google.com/drive/1nQ3n8GD6pO-ySdLlQXgbz4wA3yXoSI7i)ことで、anonymous モードの動作を確認できます。