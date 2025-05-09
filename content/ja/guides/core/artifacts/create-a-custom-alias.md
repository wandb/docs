---
title: アーティファクトのエイリアスを作成する
description: W&B アーティファクトのカスタムエイリアスを作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-custom-alias
    parent: artifacts
weight: 5
---

エイリアスを特定のバージョンへのポインターとして使用します。デフォルトでは、`Run.log_artifact` はログされたバージョンに `latest` エイリアスを追加します。

アーティファクトバージョン `v0` は、アーティファクトを初めてログする際に作成され、アーティファクトに付随します。W&B は、同じアーティファクトに再度ログを行うときにコンテンツのチェックサムを行います。アーティファクトが変更された場合、W&B は新しいバージョン `v1` を保存します。

例えば、トレーニングスクリプトがデータセットの最新バージョンを取得する場合、そのアーティファクトを使用するときに `latest` を指定します。次のコード例は、エイリアス `latest` を持つデータセットアーティファクト `bike-dataset` をダウンロードする方法を示しています。

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

アーティファクトバージョンにカスタムエイリアスを適用することもできます。例えば、モデルのチェックポイントがメトリック AP-50 で最高であることを示すために、文字列 `'best-ap50'` をエイリアスとしてモデルアーティファクトにログを記録する際に追加できます。

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```