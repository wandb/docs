---
title: 匿名モード
description: W&B アカウントなしでデータをログし、可視化する
menu:
  default:
    identifier: ja-guides-models-app-settings-page-anon
    parent: settings
weight: 80
---

誰もが簡単に実行できるようにしたいコードを公開していますか？ 匿名モードを使えば、相手は W&B アカウントを作成しなくても、あなたのコードを実行し、W&B ダッシュボードを確認し、結果を可視化できます。

匿名モードで結果をログできるようにするには、次のようにします:

```python
import wandb

wandb.init(anonymous="allow")
```

たとえば、次のコードスニペットでは、W&B で artifact を作成してログする方法を示しています:

```python
import wandb

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

[サンプルノートブックを試す](https://colab.research.google.com/drive/1nQ3n8GD6pO-ySdLlQXgbz4wA3yXoSI7i) と、匿名モードの動作を確認できます。