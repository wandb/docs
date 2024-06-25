---
description: モデルレジストリの用語と概念
displayed_sidebar: default
---


# Terms and concepts

<head>
  <title>Model Registry terms and concepts</title>
</head>

以下の用語は、W&B Model Registry の主要コンポーネントを説明します: [*model version*](#model-version), [*model artifact*](#model-artifact), および [*registered model*](#registered-model)。

## Model version
モデルバージョンは、単一のモデルチェックポイントを表します。モデルバージョンは、ある時点の実験内のモデルとそのファイルのスナップショットです。

モデルバージョンは、トレーニングされたモデルを説明するデータとメタデータの不変ディレクトリーです。W&B は、後日、モデルのアーキテクチャーと学習済みパラメータを保存（および復元）できるようにモデルバージョンにファイルを追加することを推奨します。

モデルバージョンは1つの [model artifact](#model-artifact) にのみ属します。モデルバージョンは、ゼロまたは複数の [registered models](#registered-model) に属することができます。モデルバージョンは、model artifact にログインされる順序で保存されます。同じ model artifact にログインするモデルの内容が前のモデルバージョンと異なる場合、W&B は自動的に新しいモデルバージョンを作成します。

シリアライゼーションプロセスで使用するライブラリ（例えば、[PyTorch](https://pytorch.org/tutorials/beginner/saving\_loading\_models.html) や [Keras](https://www.tensorflow.org/guide/keras/save\_and\_serialize) など）から生成されたファイルをモデルバージョンに保存します。

## Model alias

モデルエイリアスは、登録済みモデル内のモデルバージョンをセマンティックに関連する識別子で一意に識別または参照するための可変文字列です。プログラム的に使用したときにエイリアスが一意のバージョンを参照するようにするため、エイリアスを1つのバージョンにのみ割り当てることができます。これにより、エイリアスを使用してモデルの状態（例えば、チャンピオン、候補、プロダクション）をキャプチャすることができます。

「best」、「latest」、「production」、「staging」などのエイリアスを使用して、特別な目的を持つモデルバージョンをマークすることが一般的です。

例えば、モデルを作成し、それに "best" エイリアスを割り当てたとします。その特定のモデルを `run.use_model` を使って参照することができます。

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## Model tags
モデルタグは、1つ以上の registered models に属するキーワードまたはラベルです。

モデルタグを使用して、registered models をカテゴリーに分類し、Model Registry の検索バーでそれらのカテゴリーを検索します。モデルタグは Registered Model Card の上部に表示されます。MLタスク、所有チーム、または優先順位に基づいて registered models をグループ化するために使用することができます。同じモデルタグを複数の registered models に追加して、グループ化を可能にすることができます。

:::info
モデルタグは、グループ化や発見を目的として registered models に適用されるラベルであり、プログラム的にモデルバージョンを取得するための一意識別フレーズやニックネームである [model aliases](#model-alias) とは異なります。Model Registry のタスクを整理するためにタグを使用する方法については、[Organize models](./organize-models.md) を参照してください。
:::

## Model artifact
モデルアーティファクトは、[model versions](#model-version) のログで構成されたコレクションです。モデルバージョンは、モデルアーティファクトに対してログインされる順序で保存されます。

モデルアーティファクトには、1つまたは複数のモデルバージョンが含まれます。モデルバージョンがログインされていなければ、モデルアーティファクトは空であることもあります。

例えば、モデルアーティファクトを作成し、モデルトレーニング中にチェックポイントで定期的にモデルを保存するとします。それぞれのチェックポイントは独自の [model version](#model-version) に対応しています。モデルトレーニングとチェックポイント保存の間に作成されたすべてのモデルバージョンは、トレーニングスクリプトの最初に作成された同じモデルアーティファクトに保存されます。

以下の画像は、v0、v1、および v2 の3つのモデルバージョンを含むモデルアーティファクトを示しています。

![](@site/static/images/models/mr1c.png)

[ここに例のモデルアーティファクトを見る](https://wandb.ai/timssweeney/model\_management\_docs\_official\_v0/artifacts/model/mnist-zws7gt0n)。

## Registered model
登録済みモデルは、モデルバージョンへのポインター（リンク）のコレクションです。登録済みモデルを、同じ ML タスクの候補モデルの「ブックマーク」のフォルダーと考えることができます。登録済みモデルの各「ブックマーク」は、[model artifact](#model-artifact) に属する [model version](#model-version) へのポインターです。[model tags](#model-tags) を使用して登録済みモデルをグループ化することができます。

登録済みモデルは、通常、特定のモデリングユースケースやタスクの候補モデルを表します。例えば、異なる画像分類タスクのために、使用するモデルに基づいて登録済みモデルを作成することができます。「ImageClassifier-ResNet50」、「ImageClassifier-VGG16」、「DogBreedClassifier-MobileNetV2」など。モデルバージョンは、登録されたモデルにリンクされた順にバージョン番号が割り当てられます。

[ここに例の登録済みモデルを見る](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions)。