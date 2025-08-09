---
title: モデルレジストリの用語と概念
description: モデルレジストリの用語と概念
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-model-management-concepts
    parent: model-registry
weight: 2
---

以下の用語は、W&B モデルレジストリの主要なコンポーネントを説明します: [*model version*]({{< relref path="#model-version" lang="ja" >}})、[*model artifact*]({{< relref path="#model-artifact" lang="ja" >}})、および [*registered model*]({{< relref path="#registered-model" lang="ja" >}})。

## Model version
model version は 1 つのモデルチェックポイントを表します。model version は、実験内のモデルおよびそのファイルのある時点でのスナップショットです。

model version は、訓練済みモデルを記述するデータとメタデータの不変ディレクトリーです。W&B では、後からモデルのアーキテクチャーや学習済みパラメータを保存・復元できるようなファイルを model version に追加することを推奨しています。

model version は、1 つのみ [model artifact]({{< relref path="#model-artifact" lang="ja" >}}) に属します。model version は、0 個以上の [registered models]({{< relref path="#registered-model" lang="ja" >}}) に属することができます。model versions は、その model artifact に記録された順番で保存されます。W&B は、同じ model artifact にログしたモデルの内容が既存の model version と異なる場合、自動的に新しい model version を作成します。

model version には、利用しているモデリングライブラリ（例: [PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html)、[Keras](https://www.tensorflow.org/guide/keras/save_and_serialize)）が提供するシリアライズプロセスから生成されたファイルを保存します。

## Model alias

model alias は、registered model 内で model version をセマンティックに関連付けた識別子で一意に特定・参照できるようにする可変な文字列です。1 つの registered model の 1 バージョンのみにエイリアスを割り当てることができます。これはプログラムによりエイリアスがユニークなバージョンを指すためです。また、エイリアスを利用してモデルの状態（champion, candidate, production など）を示すこともできます。

一般的には、`"best"`、`"latest"`、`"production"`、`"staging"` などのエイリアスを使用して、特定の目的に model version をマーキングすることがよくあります。

たとえば、モデルを作成して `"best"` エイリアスを割り当てた場合、次のように `run.use_model` でそのモデルを参照できます。

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## Model tags
model tags は、1 つ以上の registered models に属するキーワードまたはラベルです。

model tags を使って Model Registry の検索バーで registered models をカテゴリ分け・検索することができます。model tags は Registered Model Card の上部に表示されます。たとえば ML タスク、担当チーム、優先度などで registered models をグループ化するのに使えます。同じ model tag を複数の registered models に追加することでグループ分けが可能です。

{{% alert %}}
model tags（grouping や検索性のため registered models に付与するラベル）は、[model aliases]({{< relref path="#model-alias" lang="ja" >}}) とは異なります。model aliases はプログラムから model version を取得するための一意の識別子またはニックネームです。Model Registry のタスク整理のためにタグを活用する方法については [Organize models]({{< relref path="./organize-models.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## Model artifact
model artifact は、記録された [model versions]({{< relref path="#model-version" lang="ja" >}}) のコレクションです。model versions は、その artifact にログされた順番で保存されます。

model artifact には 1 つ以上の model versions を含めることができます。まだ model version がログされていなければ空のままです。

例えば、model artifact を作成し、モデルのトレーニング中に定期的にチェックポイントを保存したとします。それぞれのチェックポイントが独自の [model version]({{< relref path="#model-version" lang="ja" >}}) となります。こうしてトレーニングとチェックポイント保存の過程で作成した全ての model versions は、トレーニングスクリプト開始時に作成した同じ model artifact に保存されます。

下記の画像は、3 つの model versions（v0、v1、v2）を含む model artifact の一例です。

{{< img src="/images/models/mr1c.png" alt="Model registry concepts" >}}

[こちらで example model artifact を見ることができます](https://wandb.ai/timssweeney/model_management_docs_official_v0/artifacts/model/mnist-zws7gt0n)。

## Registered model
registered model は model version へのポインター（リンク）の集合体です。registered model は、同じ ML タスクにおける候補モデルの "ブックマーク" を集めたフォルダーのような存在です。それぞれの"ブックマーク"は、[model artifact]({{< relref path="#model-artifact" lang="ja" >}}) に属する [model version]({{< relref path="#model-version" lang="ja" >}}) へのポインターとなります。[model tags]({{< relref path="#model-tags" lang="ja" >}}) で registered models をグループ化することも可能です。

registered models は、ある用途やタスクの候補モデルを表すことがよくあります。たとえば、画像分類タスクごとに異なるモデルを使って registered model を作成できます：`ImageClassifier-ResNet50`、`ImageClassifier-VGG16`、`DogBreedClassifier-MobileNetV2` など。model versions は、registered model にリンクされた順番でバージョン番号が付与されます。

[こちらで example Registered Model を見ることができます](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions)。