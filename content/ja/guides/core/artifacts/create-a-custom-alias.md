---
title: Create an artifact alias
description: W&B アーティファクトのためのカスタムエイリアスを作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-custom-alias
    parent: artifacts
weight: 5
---

エイリアスを使用して特定のバージョンを指し示すことができます。デフォルトでは、`Run.log_artifact` は `latest` エイリアスをログされたバージョンに追加します。

アーティファクトバージョン `v0` はアーティファクトを初めてログしたときに作成され、アタッチされます。W&B は再度同じアーティファクトにログする際にコンテンツのチェックサムを確認します。アーティファクトが変更された場合、W&B は新しいバージョン `v1` を保存します。

例えば、トレーニングスクリプトがデータセットの最新バージョンを取得するようにしたい場合、そのアーティファクトを使用する際に `latest` を指定します。以下のコード例は、`latest` というエイリアスを持つ `bike-dataset` という名前の最近のデータセットアーティファクトをダウンロードする方法を示しています：

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

また、カスタムエイリアスをアーティファクトバージョンに適用することもできます。例えば、モデルチェックポイントがメトリック AP-50 で最良であることを示したい場合、モデルアーティファクトをログする際にエイリアスとして `'best-ap50'` という文字列を追加することができます。

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```