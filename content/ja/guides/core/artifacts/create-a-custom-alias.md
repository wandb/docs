---
title: Artifacts のエイリアスを作成する
description: W&B Artifacts のカスタムエイリアスを作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-custom-alias
    parent: artifacts
weight: 5
---

エイリアスは特定のバージョンを指すポインタです。デフォルトでは、`Run.log_artifact` は ログに記録したバージョンに `latest` エイリアスを追加します。

初めて Artifact をログに記録すると、バージョン `v0` が作成されてその Artifact に付与されます。同じ Artifact を再度ログに記録すると、W&B はコンテンツのチェックサムを計算します。内容が変更されていれば、W&B は新しいバージョン `v1` を保存します。

例えば、トレーニングスクリプトで最新の Dataset を取得したい場合は、その Artifact を使用する際に `latest` を指定します。次のコード例は、`bike-dataset` という名前の Dataset Artifact の最新バージョン（エイリアス `latest`）をダウンロードする方法を示しています。

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

Artifact のバージョンにカスタムエイリアスを付けることもできます。例えば、Model のチェックポイントが AP-50 メトリックで最良であることを示したい場合、Model の Artifact をログに記録するときに `'best-ap50'` というエイリアスを追加できます。

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```