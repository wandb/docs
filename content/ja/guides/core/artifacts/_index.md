---
title: Artifacts
description: W&B アーティファクトの概要、仕組み、および開始方法。
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

W&B Artifacts を使用して、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力と出力としてデータをトラッキングし、バージョン管理します。たとえば、モデルトレーニングの run では、データセットを入力として受け取り、トレーニングされたモデルを出力として生成することがあります。ハイパーパラメーター、メタデータ、メトリクスを run にログし、アーティファクトを使用してモデルをトレーニングするために使用されるデータセットを入力としてログ、トラッキング、およびバージョン管理し、生成されたモデルのチェックポイントを出力として管理できます。

## ユースケース
あなたの ML ワークフロー全体で、[runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力と出力としてアーティファクトを使用することができます。データセット、モデル、または他のアーティファクトをプロセシングの入力として使用することができます。

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| ユースケース         | 入力                                | 出力                                   |
|------------------------|-----------------------------|------------------------------|
| モデルトレーニング           | データセット (トレーニングおよび検証データ)     | トレーニングされたモデル                |
| データセットの前処理         | データセット (生データ)          | データセット (前処理済みデータ) |
| モデルの評価           | モデル + データセット (テストデータ) | [W&B Table]({{< relref path="/guides/models/tables/" lang="ja" >}})                        |
| モデルの最適化      | モデル                       | 最適化されたモデル              |

{{% alert %}}
続くコードスニペットは順番に実行されることを意図しています。
{{% /alert %}}

## アーティファクトを作成する

4行のコードでアーティファクトを作成します:
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成する。
2. [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使用してアーティファクトオブジェクトを作成する。
3. モデルファイルやデータセットなど、一つ以上のファイルをアーティファクトオブジェクトに追加する。
4. あなたのアーティファクトを W&B にログする。

例として、続くコードスニペットは `example_artifact` という名前のアーティファクトに `dataset.h5` というファイルをログする方法を示しています:

```python
import wandb

run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="example_artifact", type="dataset")
artifact.add_file(local_path="./dataset.h5", name="training_dataset")
artifact.save()

# アーティファクトのバージョン "my_data" を dataset.h5 のデータを持つデータセットとしてログします
```

{{% alert %}}
外部オブジェクトストレージに保存されているファイルやディレクトリーへの参照の追加方法については、[外部ファイルのトラッキング]({{< relref path="./track-external-files.md" lang="ja" >}})ページを参照してください。
{{% /alert %}}

## アーティファクトをダウンロードする
[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドを使用して、あなたの run に入力としてマークしたいアーティファクトを指定します。

続くコードスニペットに続けて、この次のコードブロックは `training_dataset` アーティファクトの使用方法を示しています:

```python
artifact = run.use_artifact(
    "training_dataset:latest"
)  # "my_data" アーティファクトを使用する run オブジェクトを返します
```

これはアーティファクトオブジェクトを返します。

次に、返されたオブジェクトを使用して、アーティファクトのすべての内容をダウンロードします:

```python
datadir = (
    artifact.download()
)  # `my_data` アーティファクト全体をデフォルトのディレクトリにダウンロードします。
```

{{% alert %}}
アーティファクトを特定のディレクトリにダウンロードするために、`root` [parameter]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) にカスタムパスを渡すことができます。アーティファクトをダウンロードする別の方法や追加のパラメータについては、[アーティファクトをダウンロードして使用するガイド]({{< relref path="./download-and-use-an-artifact.md" lang="ja" >}})を参照してください。
{{% /alert %}}

## 次のステップ
* アーティファクトを[バージョン管理]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}})し、[更新]({{< relref path="./update-an-artifact.md" lang="ja" >}})する方法を学びます。
* [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})を使用して、あなたのアーティファクトに対する変更に反応して下流のワークフローをトリガーしたり、Slack チャンネルに通知する方法を学びます。
* トレーニングされたモデルを収容するスペースである[レジストリ]({{< relref path="/guides/core/registry/" lang="ja" >}})について学びます。
* [Python SDK]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) と [CLI]({{< relref path="/ref/cli/wandb-artifact/" lang="ja" >}}) リファレンスガイドを探索します。