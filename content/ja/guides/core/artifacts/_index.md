---
title: Artifacts
description: W&B Artifacts の概要、その仕組み、およびそれらの使用を開始する方法について説明します。
cascade:
- url: guides/artifacts/:filename
menu:
  default:
    identifier: ja-guides-core-artifacts-_index
    parent: core
url: guides/artifacts
weight: 1
---

{{< cta-button productLink="https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifact_fundamentals.ipynb" >}}

W&B Artifacts を使用して、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力および出力としてデータを追跡およびバージョン管理します。たとえば、モデルトレーニング run は、データセットを入力として受け取り、トレーニング済みのモデルを出力として生成する場合があります。ハイパー パラメーター、メタデータ、およびメトリクスを run に記録できます。また、artifact を使用して、モデルのトレーニングに使用されるデータセットを入力として、結果のモデル チェックポイントを別の artifact として、ログ記録、追跡、およびバージョン管理できます。

## ユースケース
Artifact は、[runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力および出力として、ML ワークフロー全体で使用できます。データセット、モデル、またはその他の artifact を、処理の入力として使用できます。

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| ユースケース               | 入力                       | 出力                       |
|------------------------|-----------------------------|------------------------------|
| モデルトレーニング         | データセット (トレーニングおよび検証データ)     | トレーニング済みモデル                |
| データセットの前処理 | データセット (未加工データ)          | データセット (前処理済みデータ) |
| モデルの評価       | モデル + データセット (テストデータ) | [W&B Table]({{< relref path="/guides/models/tables/" lang="ja" >}})                        |
| モデルの最適化     | モデル                       | 最適化されたモデル              |


{{% alert %}}
以降のコード スニペットは、順番に実行することを目的としています。
{{% /alert %}}

## artifact を作成する

4 行のコードで artifact を作成します。
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
2. [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使用して、artifact オブジェクトを作成します。
3. モデル ファイルやデータセットなど、1 つ以上のファイルを artifact オブジェクトに追加します。
4. artifact を W&B に記録します。

たとえば、次のコード スニペットは、`dataset.h5` というファイルを `example_artifact` という artifact に記録する方法を示しています。

```python
import wandb

run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="example_artifact", type="dataset")
artifact.add_file(local_path="./dataset.h5", name="training_dataset")
artifact.save()

# artifact バージョン "my_data" を dataset.h5 のデータを持つデータセットとして記録します
```

{{% alert %}}
Amazon S3 バケットなどの外部オブジェクト ストレージに保存されているファイルまたはディレクトリーへの参照を追加する方法については、[外部ファイルの追跡]({{< relref path="./track-external-files.md" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

## artifact をダウンロードする
[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドを使用して、run への入力としてマークする artifact を示します。

上記のコード スニペットに従って、次のコード ブロックは `training_dataset` artifact の使用方法を示しています。

```python
artifact = run.use_artifact(
    "training_dataset:latest"
)  # "my_data" artifact を使用して run オブジェクトを返します
```
これにより、artifact オブジェクトが返されます。

次に、返されたオブジェクトを使用して、artifact のすべてのコンテンツをダウンロードします。

```python
datadir = (
    artifact.download()
)  # `my_data` artifact 全体をデフォルトのディレクトリーにダウンロードします。
```

{{% alert %}}
カスタム パスを `root` [parameter]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) に渡して、artifact を特定のディレクトリーにダウンロードできます。artifact をダウンロードする別の方法、および追加のパラメータについては、[artifact のダウンロードと使用]({{< relref path="./download-and-use-an-artifact.md" lang="ja" >}}) のガイドを参照してください。
{{% /alert %}}


## 次のステップ
* artifact を[バージョン管理]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}})および[更新]({{< relref path="./update-an-artifact.md" lang="ja" >}})する方法を学びます。
* [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})を使用して、artifact の変更に応じてダウンストリーム ワークフローをトリガーしたり、Slack channel に通知したりする方法を学びます。
* トレーニング済みモデルを格納するスペースである[レジストリ]({{< relref path="/guides/core/registry/" lang="ja" >}})について学びます。
* [Python SDK]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) および [CLI]({{< relref path="/ref/cli/wandb-artifact/" lang="ja" >}}) リファレンス ガイドをご覧ください。
