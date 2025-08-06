---
title: モデルレジストリの用語と概念
description: モデルレジストリの用語と概念
menu:
  default:
    identifier: model-management-concepts
    parent: model-registry
weight: 2
---

以下の用語は、W&B モデルレジストリの主要な構成要素を表しています：[ *model version* ]({{< relref "#model-version" >}})、[ *model artifact* ]({{< relref "#model-artifact" >}})、および [ *registered model* ]({{< relref "#registered-model" >}}) です。

## Model version
model version は、1 つのモデル checkpoint を表します。model version は、Experiment 内でモデルおよびそのファイル群の、ある時点でのスナップショットです。

model version は、トレーニング済みモデルを説明するデータとメタデータから成る不変（immutable）のディレクトリーです。W&B では、将来モデルのアーキテクチャーや学習済みパラメータを保存・復元できるようなファイルを、model version に追加することを推奨しています。

model version は、1 つかつ 1 つだけの [model artifact]({{< relref "#model-artifact" >}}) に属します。model version は 0 個以上の [registered model]({{< relref "#registered-model" >}}) に属することができます。model version は、model artifact にログされた順番で保存されます。ログするモデル（同じ model artifact への保存）が、以前の model version と異なる内容であることを W&B が検知すると、自動的に新しい model version を作成します。

model version 内には、使用したライブラリ（例：[PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html) や [Keras](https://www.tensorflow.org/guide/keras/save_and_serialize) など）が提供するシリアライズプロセスで生成されたファイルを保存してください。

## Model alias

Model alias（エイリアス）は、登録済み model version を意味的に関連する識別子で一意に特定または参照するのに使える可変文字列です。エイリアスは 1 つの registered model の特定の version だけに割り当てられます。これはエイリアスがプログラム的に用いる際、ユニークな version を指せるようにするためです。また、champion（本番モデル）、candidate（候補モデル）、production（運用モデル）など、モデルの状態を表現するためにも利用できます。

`"best"`、`"latest"`、`"production"`、`"staging"` などをエイリアスに設定し、特別な役割を持つモデル version を示すのが一般的です。

例えば、あるモデルに `"best"` エイリアスを付与した場合、`run.use_model` を利用してその特定のモデルを参照できます。

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## Model tags
Model tag（タグ）は、1 つ以上の Registered Model に属するキーワードやラベルです。

Model tag を使うことで、Model Registry の検索バーから登録済みモデルをカテゴリごとに整理したり検索したりできます。Model tag は Registered Model Card の上部に表示されます。モデルを機械学習タスク、担当チーム、優先度などでグルーピングしたい場合に使えます。同じモデルトタグを複数の Registered Model に付与することで、グループ化できます。

{{% alert %}}
Model tag は、Registered Model にグルーピングや探索性のために付与するラベルですが、[model alias]({{< relref "#model-alias" >}}) とは異なるものです。Model alias は、特定の model version を取得するための一意な識別子またはニックネームです。Model Registry 内のタスク整理にタグを活用する方法の詳細は[モデルの整理]({{< relref "./organize-models.md" >}})をご覧ください。
{{% /alert %}}

## Model artifact
Model artifact は、[model version]({{< relref "#model-version" >}}) がログされた集合体です。model version は、model artifact に記録された順序で保存されます。

model artifact には、1 つまたはそれ以上の model version を含めることができます。まだ何も model version が記録されていない場合、空の model artifact も存在します。

例えば、model artifact を作成し、その後 model training の最中に checkpoint ごとにモデルを保存しているとしましょう。それぞれの checkpoint は独自の [model version]({{< relref "#model-version" >}}) に対応します。トレーニングスクリプト冒頭で作成した 1 つの model artifact に、トレーニングと checkpoint 保存で生成されたすべての model version が格納されます。

このあとに表示される画像は、3 つの model version（v0, v1, v2）を含む model artifact の例です。

{{< img src="/images/models/mr1c.png" alt="Model registry concepts" >}}

[model artifact の例はこちら](https://wandb.ai/timssweeney/model_management_docs_official_v0/artifacts/model/mnist-zws7gt0n)からご覧になれます。

## Registered model
Registered model は、model version へのポインター（リンク）の集合です。Registered model は、同じ機械学習タスク内の候補モデルの「ブックマークフォルダー」のようなイメージです。Registered model の各「ブックマーク」は、[model artifact]({{< relref "#model-artifact" >}}) に属する [model version]({{< relref "#model-version" >}}) へのリンクになっています。[model tags]({{< relref "#model-tags" >}}) を活用すれば、Registered model をグルーピングできます。

Registered model は、単一のユースケースやタスクに対する候補モデル群を表現することがよくあります。たとえば、画像分類タスクごとや使用モデルごとに Registered model を作成することがあります：`ImageClassifier-ResNet50`、`ImageClassifier-VGG16`、`DogBreedClassifier-MobileNetV2` などです。model version は、その Registered model にリンクされた順序通りに version 番号が割り当てられます。

[Registered Model の例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions)からご確認いただけます。