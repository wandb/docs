---
title: モデルレジストリの用語と概念
description: モデルレジストリの用語と概念
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-model-management-concepts
    parent: model-registry
weight: 2
---

以下の用語は、W&B モデルレジストリの主要な構成要素を説明します: [*model version*]({{< relref path="#model-version" lang="ja" >}})、[*model artifact*]({{< relref path="#model-artifact" lang="ja" >}})、および [*registered model*]({{< relref path="#registered-model" lang="ja" >}})。

## Model version
モデルバージョンは、単一のモデルチェックポイントを表します。モデルバージョンは、実験内のモデルとそのファイルのある時点でのスナップショットです。

モデルバージョンは、訓練されたモデルを記述するデータとメタデータの不変なディレクトリーです。W&B は、後でモデルのアーキテクチャーと学習されたパラメータを保存（および復元）できるように、ファイルをモデルバージョンに追加することを推奨しています。

モデルバージョンは、1つだけの [model artifact]({{< relref path="#model-artifact" lang="ja" >}}) に属します。モデルバージョンは、ゼロまたは複数の [registered models]({{< relref path="#registered-model" lang="ja" >}}) に属する場合があります。モデルバージョンは、model artifact にログされる順序で格納されます。同じ model artifact にログされたモデルの内容が以前のモデルバージョンと異なる場合、W&B は自動的に新しいモデルバージョンを作成します。

モデリングライブラリによって提供されるシリアライズプロセスから生成されたファイルをモデルバージョン内に保存します（例：[PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html) と [Keras](https://www.tensorflow.org/guide/keras/save_and_serialize)）。

## Model alias

モデルエイリアスは、登録されたモデル内でモデルバージョンを一意に識別または参照するための可変文字列です。登録されたモデルのバージョンにだけエイリアスを割り当てることができます。これは、エイリアスがプログラム的に使用されたとき、一意のバージョンを指す必要があるためです。エイリアスは、モデルの状態（チャンピオン、候補、プロダクション）をキャプチャするためにも使用されます。

"best"、"latest"、"production"、"staging" のようなエイリアスを使用して、特定の目的を持つモデルバージョンにマークを付けることは一般的です。

たとえば、モデルを作成し、それに "best" エイリアスを割り当てたとします。その特定のモデルを `run.use_model` で参照できます。

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## Model tags
モデルタグは、1つ以上の登録されたモデルに属するキーワードまたはラベルです。

モデルタグを使用して、登録されたモデルをカテゴリに整理し、モデルレジストリの検索バーでそれらのカテゴリを検索します。モデルタグは Registered Model Card の上部に表示されます。ML タスク、所有チーム、または優先順位に基づいて登録モデルをグループ化するために使用することもできます。同じモデルタグを複数の登録されたモデルに追加してグループ化を可能にします。

{{% alert %}}
登録されたモデルに適用されるラベルで、グループ化と発見性のために使用されるモデルタグは、[model aliases]({{< relref path="#model-alias" lang="ja" >}}) とは異なります。モデルエイリアスは、一意の識別子またはニックネームで、プログラム的にモデルバージョンを取得するために使用します。モデルレジストリでタスクを整理するためのタグの使用について詳しくは、[Organize models]({{< relref path="./organize-models.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## Model artifact
モデルアーティファクトは、ログされた [model versions]({{< relref path="#model-version" lang="ja" >}}) のコレクションです。モデルバージョンは、model artifact にログされた順序で保存されます。

モデルアーティファクトには1つ以上のモデルバージョンが含まれる場合があります。モデルバージョンがログされていない場合、モデルアーティファクトは空です。

たとえば、モデルアーティファクトを作成するとします。モデルのトレーニング中に、定期的にチェックポイントでモデルを保存します。各チェックポイントはその独自の [model version]({{< relref path="#model-version" lang="ja" >}}) に対応しています。トレーニングスクリプトの開始時に作成した同じモデルアーティファクトに、モデルトレーニング中とチェックポイント保存中に作成されたすべてのモデルバージョンが保存されます。

以下の画像は、3つのモデルバージョン v0、v1、v2 を含むモデルアーティファクトを示しています。

{{< img src="/images/models/mr1c.png" alt="" >}}

[モデルアーティファクトの例はこちら](https://wandb.ai/timssweeney/model_management_docs_official_v0/artifacts/model/mnist-zws7gt0n)をご覧ください。

## Registered model
登録モデルは、モデルバージョンへのポインタ（リンク）のコレクションです。登録モデルを、同じ ML タスク用の候補モデルの「ブックマーク」フォルダーとして考えることができます。登録モデルの各「ブックマーク」は、[model artifact]({{< relref path="#model-artifact" lang="ja" >}}) に属する [model version]({{< relref path="#model-version" lang="ja" >}}) へのポインタです。[model tags]({{< relref path="#model-tags" lang="ja" >}}) を使用して登録モデルをグループ化することができます。

登録モデルは、単一のモデリングユースケースやタスクに対する候補モデルを表すことがよくあります。たとえば、使用するモデルに基づいて異なる画像分類タスクの登録モデルを作成するかもしれません：`ImageClassifier-ResNet50`、`ImageClassifier-VGG16`、`DogBreedClassifier-MobileNetV2` など。モデルバージョンは、登録モデルにリンクされた順にバージョン番号が割り当てられます。

[登録モデルの例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions)をご覧ください。