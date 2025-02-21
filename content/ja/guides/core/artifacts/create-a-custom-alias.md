---
title: Create an artifact alias
description: W&B Artifacts 用にカスタム エイリアス を作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-custom-alias
    parent: artifacts
weight: 5
---

エイリアス を使用して、特定の バージョン へのポインタとして使用します。デフォルトでは、`Run.log_artifact` は、ログに記録された バージョン に `latest` エイリアス を追加します。

アーティファクト の バージョン `v0` は、最初に アーティファクト を ログ に記録したときに作成され、 アーティファクト にアタッチされます。W&B は、同じ アーティファクト に再度 ログ に記録する際に、コンテンツの チェックサム を計算します。アーティファクト が変更された場合、W&B は新しい バージョン `v1` を保存します。

たとえば、 トレーニング スクリプト で データセット の最新 バージョン を取得する場合は、その アーティファクト を使用するときに `latest` を指定します。次の コード例は、 エイリアス `latest` を持つ `bike-dataset` という名前の最新の データセット アーティファクト をダウンロードする方法を示しています。

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

カスタム エイリアス を アーティファクト バージョン に適用することもできます。たとえば、 モデル チェックポイント が メトリック AP-50 で最適であることを示す場合は、 モデル アーティファクト を ログ に記録するときに、文字列 `'best-ap50'` を エイリアス として追加できます。

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```
