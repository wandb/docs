---
description: W&B Artifactsの概要、その仕組み、そしてW&B Artifactsの開始方法。
slug: /guides/artifacts
displayed_sidebar: default
---

import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Artifacts

<CTAButtons productLink="https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W%26B_Artifacts.ipynb"/>

W&B Artifacts を使用して、[W&B Runs](../runs/intro.md) の入力および出力としてデータのトラッキングとバージョン管理を行います。例えば、モデルのトレーニング run はデータセットを入力として受け取り、トレーニングされたモデルを出力として生成する場合があります。ハイパーパラメーター、メタデータ、およびメトリクスを run にログとして記録することに加えて、アーティファクトを使用してモデルのトレーニングに使用したデータセットを入力として、結果として生成されたモデルのチェックポイントを出力としてログ、トラッキング、バージョン管理することもできます。

## ユースケース
アーティファクトは [runs](../runs/intro.md) の入力および出力として、ML ワークフロー全体で使用できます。データセットやモデル、さらには他のアーティファクトをプロセッシングの入力として使用できます。

![](/images/artifacts/artifacts_landing_page2.png)

| ユースケース           | 入力                       | 出力                        |
|------------------------|-----------------------------|-----------------------------|
| モデルトレーニング     | データセット (トレーニングデータと検証データ) | トレーニングされたモデル     |
| データセットの前処理   | データセット (生データ)     | データセット (前処理済みデータ) |
| モデルの評価           | モデル + データセット (テストデータ) | [W&B Table](../tables/intro.md) |
| モデルの最適化         | モデル                       | 最適化されたモデル          |

## アーティファクトの作成

4行のコードでアーティファクトを作成できます。
1. [W&B Run](../runs/intro.md) を作成します。
2. [`wandb.Artifact`](../../ref/python/artifact.md) API を使用してアーティファクトオブジェクトを作成します。
3. モデルファイルやデータセットなどの1つ以上のファイルをアーティファクトオブジェクトに追加します。この例では、単一ファイルを追加します。
4. W&B にアーティファクトをログします。

```python
run = wandb.init(project = "artifacts-example", job_type = "add-dataset")
run.log_artifact(data = "./dataset.h5", name = "my_data", type = "dataset" ) # dataset.h5 からデータを持つデータセットとしてアーティファクトバージョン "my_data" をログします。
```

:::tip
Amazon S3 バケットのような外部オブジェクトストレージに保存されているファイルまたはディレクトリーへの参照を追加する方法については、[外部ファイルのトラッキング](./track-external-files.md) ページをご覧ください。
:::

## アーティファクトのダウンロード
[`use_artifact`](../../ref/python/run.md#use_artifact) メソッドを使用して、run に入力とするアーティファクトを指定します。このメソッドはアーティファクトオブジェクトを返します。

```python
artifact = run.use_artifact("my_data:latest") # "my_data" アーティファクトを使用する run オブジェクトを返します。
```

次に、返されたオブジェクトを使用してアーティファクトの全内容をダウンロードします。

```python
datadir = artifact.download() # デフォルトのディレクトリーに "my_data" アーティファクト全体をダウンロードします。
```

:::tip
カスタムパスを `root` [パラメータ](../../ref/python/artifact.md) に渡して、特定のディレクトリーにアーティファクトをダウンロードできます。アーティファクトをダウンロードする他の方法および追加のパラメーターについては、[アーティファクトのダウンロードと使用](./download-and-use-an-artifact.md) に関するガイドを参照してください。
:::

## 次のステップ
* アーティファクトの[バージョン管理](./create-a-new-artifact-version.md)、[更新](./update-an-artifact.md)、または [削除](./delete-artifacts.md) について学びます。
* アーティファクトの変更に応じて下流のワークフローをトリガーする方法を [artifact automation](./project-scoped-automations.md) で学びます。
* トレーニング済みモデルを収容するスペースである [model registry](../model_registry/intro.md) について学びます。
* [Python SDK](../../ref/python/artifact.md) および [CLI](../../ref/cli/wandb-artifact/README.md) リファレンスガイドを探索します。