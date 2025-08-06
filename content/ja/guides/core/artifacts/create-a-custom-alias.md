---
title: アーティファクトのエイリアスを作成する
description: W&B Artifacts にカスタムエイリアスを作成する
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-custom-alias
    parent: artifacts
weight: 5
---

エイリアスは特定のバージョンへのポインタとして利用できます。デフォルトでは、`Run.log_artifact` はログに記録したバージョンに `latest` エイリアスを自動的に追加します。

初めて artifact をログした際、artifact バージョン `v0` が作成され、artifact に付与されます。同じ artifact に再度ログする際に、W&B は内容のチェックサムを計算します。内容が変更されていれば、新しいバージョン `v1` を保存します。

例えば、トレーニングスクリプトで常に最新のデータセットバージョンを取得したい場合、その artifact を使用するときに `latest` を指定します。以下のコード例は、`latest` というエイリアスを持つ `bike-dataset` という名前の dataset artifact をダウンロードする方法を示しています。

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

カスタムエイリアスを artifact バージョンに付与することもできます。例えば、あるモデルのチェックポイントが指標 AP-50 において最良であることを示したい場合、モデル artifact をログする際に `'best-ap50'` という文字列をエイリアスとして追加できます。

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```