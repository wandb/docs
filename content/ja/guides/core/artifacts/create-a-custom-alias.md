---
title: アーティファクトのエイリアスを作成する
description: W&B Artifacts にカスタムエイリアスを作成する
menu:
  default:
    identifier: create-a-custom-alias
    parent: artifacts
weight: 5
---

エイリアスを特定のバージョンへのポインタとして使用します。デフォルトでは、`Run.log_artifact` がログされたバージョンに `latest` エイリアスを追加します。

初めてアーティファクトをログする際、そのアーティファクトにバージョン `v0` が作成され、付与されます。同じアーティファクトに再度ログした場合、W&B は中身のチェックサムを計算します。もし内容が変更されていれば、新しいバージョン `v1` として保存されます。

例えば、トレーニングスクリプトで一番新しいバージョンのデータセットを取得したい場合、そのアーティファクトを使用するときに `latest` を指定します。以下のコード例は、`latest` というエイリアスが付けられた `bike-dataset` という名前の最近のデータセットアーティファクトをダウンロードする方法を示しています。

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

また、アーティファクトのバージョンにカスタムエイリアスを付与することもできます。例えば、あるモデルのチェックポイントが AP-50 というメトリクスで最高であることを示したい場合、モデルアーティファクトをログする際に `'best-ap50'` という文字列をエイリアスとして追加できます。

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```