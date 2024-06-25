---
description: W&B Artifacts のカスタムエイリアスを作成する。
displayed_sidebar: default
---


# カスタムエイリアスを作成する

<head>
    <title>Create a custom alias for your Artifact.</title>
</head>

エイリアスは特定のバージョンへのポインタとして使用します。デフォルトでは、`Run.log_artifact` はログされたバージョンに `latest` エイリアスを追加します。

初めてアーティファクトをログすると、アーティファクトバージョン `v0` が作成され、それがアーティファクトに添付されます。W&Bは同じアーティファクトに再度ログすると内容をチェックサムし、アーティファクトが変更されている場合、新しいバージョン `v1` を保存します。

例えば、トレーニングスクリプトで最新バージョンのデータセットを取得したい場合、そのアーティファクトを使用するときに `latest` を指定します。以下のコード例は、エイリアス `latest` を持つ `bike-dataset` という名前の最近のデータセットArtifactをダウンロードする方法を示しています。

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

また、カスタムエイリアスをアーティファクトバージョンに適用することもできます。例えば、モデルチェックポイントがAP-50のメトリックで最高であることをマークしたい場合、モデルアーティファクトをログするときにエイリアスとして文字列 `'best-ap50'` を追加できます。

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```
