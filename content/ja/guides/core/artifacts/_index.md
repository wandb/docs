---
title: Artifacts
description: W&B Artifacts の概要、その仕組み、および始め方について説明します。
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

W&B Artifacts を使用して、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力および出力としてデータを追跡し、バージョン管理します。例えば、モデル トレーニングの Run はデータセットを入力として受け取り、学習済みモデルを出力として生成する場合があります。Run にハイパーパラメーター、メタデータ、およびメトリクスをログでき、Artifact を使用して、モデルのトレーニングに使用されたデータセットを入力としてログ、追跡、バージョン管理し、結果として得られるモデルのチェックポイントを別の Artifact として出力できます。

## ユースケース
Artifacts は、ML ワークフロー全体を通じて [Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力および出力として使用できます。データセット、モデル、またはその他の Artifact でさえも、前処理の入力として使用できます。

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| ユースケース               | 入力                       | 出力                       |
|------------------------|-----------------------------|------------------------------|
| モデルトレーニング         | データセット (トレーニングおよび検証データ)     | 学習済みモデル                |
| データセットのプロセッシング | データセット (生データ)          | データセット (プロセッシング済みデータ) |
| モデルの評価       | モデル + データセット (テストデータ) | [W&B テーブル]({{< relref path="/guides/models/tables/" lang="ja" >}})                        |
| モデルの最適化     | モデル                       | 最適化されたモデル              |


{{% alert %}}
以下のコードスニペットは、順番に実行されることを意図しています。
{{% /alert %}}

## Artifact の作成

Artifact は 4 行のコードで作成できます。
1. [W&B Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
2. [`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) API を使用して Artifact オブジェクトを作成します。
3. モデルファイルやデータセットなど、1 つまたは複数のファイルを Artifact オブジェクトに追加します。
4. Artifact を W&B にログします。

例えば、以下のコードスニペットは、`dataset.h5` という名前のファイルを `example_artifact` という Artifact にログする方法を示しています。

```python
import wandb

run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="example_artifact", type="dataset")
artifact.add_file(local_path="./dataset.h5", name="training_dataset")
artifact.save()

# Artifact バージョン "my_data" を dataset.h5 のデータを持つデータセットとしてログします。
```

- `type` は、W&B プラットフォームでの Artifact の表示方法に影響します。`type` を指定しない場合、デフォルトは `unspecified` です。
- ドロップダウンの各ラベルは異なる `type` パラメータ値を表します。上記のコードスニペットでは、Artifact の `type` は `dataset` です。

{{% alert %}}
Amazon S3 バケットのような外部オブジェクトストレージに保存されているファイルまたはディレクトリーへの参照を追加する方法については、[外部ファイルの追跡]({{< relref path="./track-external-files.md" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

## Artifact のダウンロード
Run への入力としてマークする Artifact を [`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) メソッドで指定します。

上記のコードスニペットに続いて、この次のコードブロックは、`training_dataset` Artifact を使用する方法を示しています。

```python
artifact = run.use_artifact(
    "training_dataset:latest"
)  # "my_data" を使用する Artifact を返します
```
これにより Artifact オブジェクトが返されます。

次に、返されたオブジェクトを使用して Artifact のすべてのコンテンツをダウンロードします。

```python
datadir = (
    artifact.download()
)  # `my_data` Artifact 全体をデフォルトのディレクトリにダウンロードします。
```

{{% alert %}}
`root` [パラメータ]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) にカスタムパスを渡して、Artifact を特定のディレクトリーにダウンロードできます。Artifact をダウンロードする別の方法や追加のパラメータについては、[Artifact のダウンロードと使用]({{< relref path="./download-and-use-an-artifact.md" lang="ja" >}}) のガイドを参照してください。
{{% /alert %}}

## 次のステップ
* Artifact の [バージョン管理]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}}) および [更新]({{< relref path="./update-an-artifact.md" lang="ja" >}}) の方法を学習します。
* Artifact の変更に応じて、[オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) を使用してダウンストリームのワークフローをトリガーしたり、Slack チャンネルに通知したりする方法を学習します。
* 学習済みモデルを格納するスペースである [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) について学習します。
* [Python SDK]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) および [CLI]({{< relref path="/ref/cli/wandb-artifact/" lang="ja" >}}) のリファレンスガイドを確認します。