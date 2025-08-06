---
title: Artifacts
description: W&B Artifacts の概要、仕組み、そして基本的な使い方について説明します。
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

W&B Artifacts を使うことで、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入出力として データの追跡やバージョン管理ができます。例えば、モデルのトレーニング run ではデータセットを入力として使い、トレーニング済みモデルを出力として生成します。run にはハイパーパラメーター、メタデータ、メトリクスを記録でき、アーティファクトを使うことで、トレーニングに利用したデータセットのバージョン管理や記録、さらに出力されたモデルのチェックポイントを別のアーティファクトとして管理することができます。

## ユースケース
Artifacts は、ML ワークフロー全体を通して [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入出力として活用できます。データセット、モデル、または他の artifacts をプロセッシングの入力として利用可能です。

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| ユースケース           | 入力                                  | 出力                             |
|------------------------|---------------------------------------|----------------------------------|
| モデルトレーニング     | データセット（トレーニング・検証データ） | トレーニング済みモデル            |
| データセット前処理     | データセット（生データ）               | データセット（前処理済みデータ）  |
| モデルの評価           | モデル + データセット（テストデータ）    | [W&B Table]({{< relref path="/guides/models/tables/" lang="ja" >}}) |
| モデル最適化           | モデル                                 | 最適化済みモデル                  |


{{% alert %}}
この後に出てくるコードスニペットは、順番に実行することを想定しています。
{{% /alert %}}

## アーティファクトの作成

アーティファクトは以下の4ステップで作成できます。
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成する
2. [`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) API を使って artifact オブジェクトを作成する
3. モデルファイルやデータセットなどのファイルを artifact オブジェクトに追加する
4. アーティファクトを W&B にログする

例えば、次のコードスニペットは `dataset.h5` というファイルを `example_artifact` というアーティファクトにログする方法です。

```python
import wandb

run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="example_artifact", type="dataset")
artifact.add_file(local_path="./dataset.h5", name="training_dataset")
artifact.save()

# アーティファクトバージョン "my_data" を、dataset.h5 のデータを持つ dataset として記録します
```

- アーティファクトの `type` を指定することで、W&B プラットフォーム上の表示のされ方が変わります。`type` を明示しない場合は `unspecified` になります。
- ドロップダウンの各ラベルは、それぞれ異なる `type` パラメータ値を表します。上記コードスニペットでは、artifact の `type` は `dataset` です。

{{% alert %}}
外部オブジェクトストレージ（例: Amazon S3 バケット）に保存されたファイルやディレクトリーを参照として追加する方法については、[外部ファイルのトラッキング]({{< relref path="./track-external-files.md" lang="ja" >}}) ページをご覧ください。
{{% /alert %}}

## アーティファクトのダウンロード
run への入力としてマークしたいアーティファクトは、[`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) メソッドで指定できます。

上のコードスニペットに続けて、`training_dataset` アーティファクトを使う例は以下の通りです。

```python
artifact = run.use_artifact(
    "training_dataset:latest"
)  # "my_data" アーティファクトを利用する run オブジェクトが返ります
```
この処理で artifact オブジェクトが返されます。

次に、そのオブジェクトを使ってアーティファクトの全内容をダウンロードします。

```python
datadir = (
    artifact.download()
)  # デフォルトディレクトリに `my_data` アーティファクト全体をダウンロードします
```

{{% alert %}}
アーティファクトを特定のディレクトリにダウンロードしたい場合は、`root` [パラメータ]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) に任意のパスを渡してください。その他のダウンロード方法や追加のパラメータについては、[アーティファクトのダウンロードと利用方法]({{< relref path="./download-and-use-an-artifact.md" lang="ja" >}}) ガイドをご覧ください。
{{% /alert %}}


## 次のステップ
* アーティファクトの[バージョン管理]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}})や[アップデート]({{< relref path="./update-an-artifact.md" lang="ja" >}})方法を学ぶ
* [automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) でアーティファクトの変更時に下流ワークフローの自動実行や Slack チャンネルへの通知方法を学ぶ
* 学習済みモデルのためのスペースである [registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) について知る
* [Python SDK]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) や [CLI]({{< relref path="/ref/cli/wandb-artifact/" lang="ja" >}}) リファレンスガイドを探ってみる