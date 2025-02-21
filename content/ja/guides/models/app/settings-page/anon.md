---
title: Anonymous mode
description: W&B アカウントなしで データ を ログ し、可視化する
menu:
  default:
    identifier: ja-guides-models-app-settings-page-anon
    parent: settings
weight: 80
---

誰でも簡単に実行できる コード を公開したいですか？ 匿名モードを使用すると、W&B の アカウント を最初に作成しなくても、他の人があなたの コード を実行し、W&B の ダッシュボード を見て、 結果 を可視化できます。

匿名モードで 結果 を ログ に記録できるようにするには、次のようにします。

```python
import wandb

wandb.init(anonymous="allow")
```

たとえば、次の コードスニペット は、W&B で アーティファクト を作成して ログ に記録する方法を示しています。

```python
import wandb

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

匿名モードの動作を確認するには、[ノートブック の例](https://colab.research.google.com/drive/1nQ3n8GD6pO-ySdLlQXgbz4wA3yXoSI7i) をお試しください。
