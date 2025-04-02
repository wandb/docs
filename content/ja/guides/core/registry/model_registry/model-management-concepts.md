---
title: Model Registry Terms and Concepts
description: モデルレジストリ の用語と概念
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-model-management-concepts
    parent: model-registry
weight: 2
---

W&B の モデルレジストリ の主要コンポーネントは、以下の用語で説明されます。[*モデル バージョン*]({{< relref path="#model-version" lang="ja" >}}), [*モデル artifact*]({{< relref path="#model-artifact" lang="ja" >}}), そして [*registered model*]({{< relref path="#registered-model" lang="ja" >}})。

## モデル バージョン
モデル バージョンは、単一のモデル チェックポイント を表します。モデル バージョンは、実験 内におけるモデルとそのファイルの特定時点での スナップショット です。

モデル バージョンは、トレーニング されたモデルを記述するデータと メタデータ の不変の ディレクトリー です。W&B は、モデルの アーキテクチャー と学習済み パラメータ を後で保存（および復元）できるように、モデル バージョンにファイルを追加することをお勧めします。

モデル バージョンは、1つの [モデル artifact]({{< relref path="#model-artifact" lang="ja" >}}) にのみ属します。モデル バージョンは、ゼロまたは複数の [registered model]({{< relref path="#registered-model" lang="ja" >}}) に属することができます。モデル バージョンは、モデル artifact に ログ された順にモデル artifact に保存されます。W&B は、(同じモデル artifact に) ログ するモデルの内容が以前のモデル バージョンと異なる場合、新しいモデル バージョンを自動的に作成します。

モデリング ライブラリ (たとえば、[PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html) や [Keras](https://www.tensorflow.org/guide/keras/save_and_serialize)) によって提供されるシリアル化 プロセス から生成されたファイルをモデル バージョン内に保存します。

## モデル エイリアス

モデル エイリアス は、 registered model 内のモデル バージョンを、セマンティックに関連する識別子で一意に識別または参照できる、変更可能な文字列です。エイリアス は、 registered model の 1 つの バージョン にのみ割り当てることができます。これは、 エイリアス がプログラムで使用される場合に一意の バージョン を参照する必要があるためです。また、 エイリアス を使用してモデルの状態 (チャンピオン、候補、本番) をキャプチャすることもできます。

`"best"`、`"latest"`、`"production"`、`"staging"` などの エイリアス を使用して、特別な目的を持つモデル バージョン をマークするのが一般的な方法です。

たとえば、モデルを作成し、それに `"best"` エイリアス を割り当てるとします。`run.use_model` でその特定のモデルを参照できます。

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## モデル タグ
モデル タグ は、1 つまたは複数の registered model に属する キーワード または ラベル です。

モデル タグ を使用して、 registered model を カテゴリ に整理し、 モデルレジストリ の検索バーでそれらの カテゴリ を検索します。モデル タグ は、 Registered Model Card の上部に表示されます。これらを使用して、 registered model を ML タスク 、所有 チーム 、または 優先度 でグループ化することもできます。同じモデル タグ を複数の registered model に追加して、グループ化できます。

{{% alert %}}
モデル タグ は、グループ化と発見可能性のために registered model に適用される ラベル であり、[モデル エイリアス]({{< relref path="#model-alias" lang="ja" >}}) とは異なります。モデル エイリアス は、プログラムでモデル バージョン を取得するために使用する一意の識別子または ニックネーム です。タグ を使用して モデルレジストリ 内の タスク を整理する方法の詳細については、[モデル の整理]({{< relref path="./organize-models.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## モデル artifact
モデル artifact は、 ログ された [モデル バージョン]({{< relref path="#model-version" lang="ja" >}}) のコレクションです。モデル バージョンは、モデル artifact に ログ された順にモデル artifact に保存されます。

モデル artifact には、1 つ以上のモデル バージョン を含めることができます。モデル バージョン が ログ されていない場合、モデル artifact は空になる可能性があります。

たとえば、モデル artifact を作成するとします。モデル トレーニング 中に、 チェックポイント 中にモデルを定期的に保存します。各 チェックポイント は、独自の [モデル バージョン]({{< relref path="#model-version" lang="ja" >}}) に対応します。モデル トレーニング 中に作成され、 チェックポイント の保存されたすべてのモデル バージョン は、 トレーニング スクリプト の開始時に作成した同じモデル artifact に保存されます。

次の図は、v0、v1、v2 の 3 つのモデル バージョン を含むモデル artifact を示しています。

{{< img src="/images/models/mr1c.png" alt="" >}}

[モデル artifact の例はこちら](https://wandb.ai/timssweeney/model_management_docs_official_v0/artifacts/model/mnist-zws7gt0n) でご覧ください。

## Registered model
Registered model は、モデル バージョン への ポインタ (リンク) のコレクションです。registered model は、同じ ML タスク の候補モデルの「 ブックマーク 」の フォルダー と考えることができます。registered model の各「 ブックマーク 」は、[モデル artifact]({{< relref path="#model-artifact" lang="ja" >}}) に属する [モデル バージョン]({{< relref path="#model-version" lang="ja" >}}) への ポインタ です。[モデル タグ]({{< relref path="#model-tags" lang="ja" >}}) を使用して、 registered model をグループ化できます。

Registered model は、多くの場合、単一のモデリング ユースケース または タスク の候補モデルを表します。たとえば、使用するモデルに基づいて、さまざまな 画像分類 タスク の registered model を作成する場合があります:`ImageClassifier-ResNet50`、`ImageClassifier-VGG16`、`DogBreedClassifier-MobileNetV2` など。モデル バージョン には、 registered model にリンクされた順に バージョン 番号が割り当てられます。

[Registered Model の例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions) でご覧ください。
