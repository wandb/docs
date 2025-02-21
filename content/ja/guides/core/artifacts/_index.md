---
title: Artifacts
description: W&B アーティファクトの概要、動作の仕組み、W&B アーティファクトの開始方法。
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

W&B の Artifacts を使用して、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力と出力としてデータを追跡およびバージョン管理します。例えば、モデルのトレーニングの run は、データセットを入力として受け取り、トレーニングされたモデルを出力として生成することができます。run にハイパーパラメータ、メタデータ、メトリクスをログし、アーティファクトを使用して、モデルのトレーニングに使用されたデータセットを入力としてログ、追跡、バージョン管理し、結果として得られるモデルのチェックポイントを出力として別のアーティファクトにログすることができます。

## ユースケース
アーティファクトは、ML ワークフロー全体を通じて、[runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の入力と出力として使用できます。データセット、モデル、または他のアーティファクトをプロセッシングの入力として使用できます。

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| ユースケース              | 入力                        | 出力                         |
|------------------------|-----------------------------|------------------------------|
| モデルのトレーニング       | データセット (トレーニングおよび検証データ)    | トレーニングされたモデル          |
| データセットの前処理          | データセット (生データ)             | データセット (前処理されたデータ) |
| モデルの評価               | モデル + データセット (テストデータ) | [W&B Table]({{< relref path="/guides/core/tables/" lang="ja" >}}) |
| モデルの最適化             | モデル                        | 最適化されたモデル               |

{{% alert %}}
次のコードスニペットは順番に実行することを意図しています。
{{% /alert %}}

## アーティファクトの作成

4 行のコードでアーティファクトを作成します：
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成します。
2. [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API でアーティファクトオブジェクトを作成します。
3. モデルファイルやデータセットなど、1 つ以上のファイルをあなたのアーティファクトオブジェクトに追加します。
4. W&B でアーティファクトをログします。

例えば、次のコードスニペットは、`dataset.h5` というファイルを `example_artifact` というアーティファクトにログする方法を示しています：

```python
import wandb

run = wandb.init(project = "artifacts-example", job_type = "add-dataset")
artifact = wandb.Artifact(name = "example_artifact", type = "dataset")
artifact.add_file(local_path = "./dataset.h5", name = "training_dataset")
artifact.save()

# アーティファクトのバージョン "my_data" を dataset.h5 からのデータとしてデータセットとしてログします
```

{{% alert %}}
外部のオブジェクトストレージに保存されたファイルやディレクトリーへの参照を追加する方法については、[外部ファイルの追跡]({{< relref path="./track-external-files.md" lang="ja" >}}) ページを参照してください。 
{{% /alert %}}

## アーティファクトのダウンロード
[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドで run の入力としてマークしたいアーティファクトを示します。

前のコードスニペットに続いて、次のコードブロックでは `training_dataset` アーティファクトを使用する方法を示しています：

```python
artifact = run.use_artifact("training_dataset:latest") # "my_data" アーティファクトを使用する run オブジェクトを返します
```
これはアーティファクトオブジェクトを返します。

次に、返されたオブジェクトを使用して、アーティファクトのすべての内容をダウンロードします：

```python
datadir = artifact.download() # デフォルトディレクトリーに "my_data" アーティファクトを完全にダウンロードします。
```

{{% alert %}}
アーティファクトを特定のディレクトリーにダウンロードするために `root` [パラメータ]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) にカスタムパスを渡すことができます。アーティファクトをダウンロードする別の方法や追加のパラメータを見るためには、[アーティファクトのダウンロードと使用]({{< relref path="./download-and-use-an-artifact.md" lang="ja" >}}) のガイドを参照してください。
{{% /alert %}}

## 次のステップ
* アーティファクトをどのように[バージョン管理]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}})し、[更新]({{< relref path="./update-an-artifact.md" lang="ja" >}})するかを学びます。
* アーティファクトの変更に応じて下流のワークフローをトリガーする方法を[アーティファクトオートメーション]({{< relref path="/guides/models/automations/project-scoped-automations/" lang="ja" >}})で学びます。
* トレーニングされたモデルを収容するスペースである[レジストリ]({{< relref path="/guides/models/registry/" lang="ja" >}})について学びます。
* [Python SDK]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) と [CLI]({{< relref path="/ref/cli/wandb-artifact/" lang="ja" >}}) のリファレンスガイドを探ります。