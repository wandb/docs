---
title: Artifacts
description: W&B Artifacts の概要、その仕組み、そして開始方法について説明します。
menu:
  default:
    identifier: artifacts
    parent: core
url: guides/artifacts
cascade:
- url: guides/artifacts/:filename
weight: 1
---

{{< cta-button productLink="https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifact_fundamentals.ipynb" >}}

W&B Artifacts を使って、データを [W&B Runs]({{< relref "/guides/models/track/runs/" >}}) の入出力として管理・バージョン管理しましょう。例えば、モデルのトレーニング run はデータセットを入力として受け取り、学習済みモデルを出力として生成します。ハイパーパラメーター、メタデータ、メトリクスを run にログできるだけでなく、アーティファクトを使うことで、モデルの学習に使ったデータセットや、出力されたモデルのチェックポイントを、それぞれ入力・出力として記録・追跡・バージョン管理できます。

## ユースケース
Artifacts は、[runs]({{< relref "/guides/models/track/runs/" >}}) の入出力として、ML ワークフロー全体で活用できます。データセットやモデル、または他の artifacts を入力として、さまざまなプロセシングに使用できます。

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| ユースケース                 | 入力                                  | 出力                                   |
|------------------------------|---------------------------------------|----------------------------------------|
| モデルトレーニング           | データセット（トレーニング・検証用）  | 学習済みモデル                         |
| データセット前処理           | データセット（生データ）              | データセット（前処理済みデータ）       |
| モデルの評価                 | モデル + データセット（テストデータ） | [W&B Table]({{< relref "/guides/models/tables/" >}})                 |
| モデルの最適化               | モデル                                | 最適化済みモデル                       |


{{% alert %}}
以下のコードスニペットは順番に実行することを想定しています。
{{% /alert %}}

## アーティファクトの作成

4 行のコードで artifact を作成できます。
1. [W&B run]({{< relref "/guides/models/track/runs/" >}}) を作成します。
2. [`wandb.Artifact`]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) API でアーティファクトオブジェクトを作成します。
3. モデルファイルやデータセットなど、ファイルをアーティファクトオブジェクトに追加します。
4. アーティファクトを W&B にログします。

例として、`dataset.h5` というファイルを `example_artifact` というアーティファクトにログするコードスニペットです。

```python
import wandb

run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="example_artifact", type="dataset")
artifact.add_file(local_path="./dataset.h5", name="training_dataset")
artifact.save()

# アーティファクトバージョン "my_data" を dataset.h5 のデータで "dataset" としてログします
```

- artifact の `type` は W&B プラットフォームでの表示方法に影響します。`type` を指定しない場合は `"unspecified"` となります。
- ドロップダウンの各ラベルが、異なる `type` パラメーター値を表します。上記コードスニペットでは、artifact の `type` は `dataset` です。

{{% alert %}}
外部のオブジェクトストレージ（Amazon S3 バケットなど）に保存されたファイルやディレクトリへの参照を追加する方法は、[外部ファイルのトラッキング]({{< relref "./track-external-files.md" >}}) をご参照ください。
{{% /alert %}}

## アーティファクトをダウンロードする
run の入力にしたいアーティファクトを、[`use_artifact`]({{< relref "/ref/python/sdk/classes/run.md#use_artifact" >}}) メソッドで指定します。

前述のコードスニペットに続けて、このように `training_dataset` アーティファクトを利用します。

```python
artifact = run.use_artifact(
    "training_dataset:latest"
)  # "my_data" アーティファクトを使う run オブジェクトを返します
```
これで artifact オブジェクトが返ります。

次に、このオブジェクトを使って、artifact の内容をすべてダウンロードします。

```python
datadir = (
    artifact.download()
)  # `my_data` artifact をデフォルトのディレクトリに全てダウンロードします
```

{{% alert %}}
任意のディレクトリに artifact をダウンロードしたい場合は、`root` [パラメータ]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) を指定できます。他のダウンロード方法や追加パラメータについては、[アーティファクトのダウンロードと利用]({{< relref "./download-and-use-an-artifact.md" >}}) ガイドを参照してください。
{{% /alert %}}


## 次のステップ
* アーティファクトの[バージョン管理]({{< relref "./create-a-new-artifact-version.md" >}})や[更新]({{< relref "./update-an-artifact.md" >}})方法を学ぶ
* [automations]({{< relref "/guides/core/automations/" >}}) を使い、artifacts の変更に応じて後続ワークフローのトリガーや Slack チャンネルへの通知方法を学ぶ
* 学習済みモデルをまとめて管理できる [registry]({{< relref "/guides/core/registry/" >}}) について知る
* [Python SDK]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) や [CLI]({{< relref "/ref/cli/wandb-artifact/" >}}) リファレンスガイドを読む