---
title: Artifacts
description: W&B Artifacts の概要、その仕組み、そして W&B Artifacts の開始方法について説明します。
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

W&B Artifacts を使用して、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力および出力としてデータを追跡およびバージョン管理します。たとえば、モデルトレーニング run は、データセットを入力として受け取り、トレーニング済みのモデルを出力として生成する場合があります。ハイパーパラメーター 、メタデータ、および メトリクス を run に ログ できます。また、artifact を使用して、モデルのトレーニングに使用するデータセットを入力として、および結果のモデル チェックポイント を出力として ログ 、追跡、およびバージョン管理できます。

## ユースケース
[runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力および出力として、ML ワークフロー 全体で Artifacts を使用できます。データセット、モデル、またはその他の artifact を、プロセッシング の入力として使用できます。

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| ユースケース               | 入力                       | 出力                       |
|------------------------|-----------------------------|------------------------------|
| モデルトレーニング         | データセット (トレーニング および 検証 データ)     | トレーニング済み モデル                |
| データセット の事前処理 | データセット (raw データ)          | データセット (事前処理済み データ) |
| モデル評価       | モデル + データセット (テスト データ) | [W&B Table]({{< relref path="/guides/core/tables/" lang="ja" >}})                        |
| モデル最適化     | モデル                       | 最適化されたモデル              |


{{% alert %}}
次の コードスニペット は、順番に実行されることを意図しています。
{{% /alert %}}

## artifact の作成

4 行の コード で artifact を作成します。
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
2. [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使用して、artifact オブジェクトを作成します。
3. モデル ファイル や データセット など、1 つまたは複数のファイルを artifact オブジェクトに追加します。
4. artifact を W&B に ログ します。

たとえば、次の コードスニペット は、`dataset.h5` というファイルを `example_artifact` という artifact に ログ する方法を示しています。

```python
import wandb

run = wandb.init(project = "artifacts-example", job_type = "add-dataset")
artifact = wandb.Artifact(name = "example_artifact", type = "dataset")
artifact.add_file(local_path = "./dataset.h5", name = "training_dataset")
artifact.save()

# artifact のバージョン "my_data" を dataset.h5 のデータを持つデータセットとしてログします。
```

{{% alert %}}
Amazon S3 バケット など、外部オブジェクト ストレージに保存されているファイルまたは ディレクトリー への参照を追加する方法については、[track external files]({{< relref path="./track-external-files.md" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

## artifact のダウンロード
[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドを使用して、run への入力としてマークする artifact を指定します。

上記の コードスニペット に続いて、次の コード ブロック は `training_dataset` artifact の使用方法を示しています。

```python
artifact = run.use_artifact("training_dataset:latest") # "my_data" artifact を使用して run オブジェクトを返します
```
これにより、artifact オブジェクトが返されます。

次に、返されたオブジェクトを使用して、artifact のすべてのコンテンツをダウンロードします。

```python
datadir = artifact.download() # "my_data" artifact 全体をデフォルト ディレクトリー にダウンロードします。
```

{{% alert %}}
`root` [parameter]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) にカスタム パスを渡して、artifact を特定の ディレクトリー にダウンロードできます。artifact をダウンロードする別の方法、および追加の parameter を確認するには、[downloading and using artifacts]({{< relref path="./download-and-use-an-artifact.md" lang="ja" >}}) の ガイド を参照してください。
{{% /alert %}}


## 次のステップ
* artifact の[バージョン管理]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}}) および[更新]({{< relref path="./update-an-artifact.md" lang="ja" >}}) 方法を学びます。
* [artifact automation]({{< relref path="/guides/models/automations/project-scoped-automations/" lang="ja" >}}) を使用して、artifact への変更に応じてダウンストリーム ワークフロー をトリガーする方法を学びます。
* トレーニング済み モデル を格納するスペースである[registry]({{< relref path="/guides/models/registry/" lang="ja" >}}) について学びます。
* [Python SDK]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) および [CLI]({{< relref path="/ref/cli/wandb-artifact/" lang="ja" >}}) リファレンス ガイド を参照してください。
