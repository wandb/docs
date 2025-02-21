---
title: Model Registry Terms and Concepts
description: モデルレジストリ の用語と概念
menu:
  default:
    identifier: ja-guides-models-registry-model_registry-model-management-concepts
    parent: model-registry
weight: 2
---

W&B モデルレジストリ の主要なコンポーネントは、[*モデル バージョン*]({{< relref path="#model-version" lang="ja" >}})、[*モデル artifact*]({{< relref path="#model-artifact" lang="ja" >}})、および [*登録済みモデル*]({{< relref path="#registered-model" lang="ja" >}}) です。

## モデル バージョン
モデル バージョンは、単一のモデル チェックポイントを表します。モデル バージョンは、実験におけるモデルとそのファイルの特定時点でのスナップショットです。

モデル バージョンは、トレーニング されたモデルを記述するデータとメタデータの不変のディレクトリー です。W&B は、モデル アーキテクチャー と学習済み パラメータ を後で保存 (および復元) できるように、ファイルをモデル バージョンに追加することをお勧めします。

モデル バージョンは、1 つだけ、[モデル artifact]({{< relref path="#model-artifact" lang="ja" >}}) に属します。モデル バージョンは、ゼロまたは複数の [登録済みモデル]({{< relref path="#registered-model" lang="ja" >}}) に属することができます。モデル バージョンは、モデル artifact にログ された順にモデル artifact に保存されます。W&B は、(同じモデル artifact に) ログ するモデルの内容が以前のモデル バージョンと異なることを検出した場合、新しいモデル バージョンを自動的に作成します。

モデル ライブラリ が提供するシリアル化 プロセス から生成されたファイルをモデル バージョン内に保存します (たとえば、[PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html) や [Keras](https://www.tensorflow.org/guide/keras/save_and_serialize) など)。

## モデル エイリアス

モデル エイリアス は、登録済みモデル内のモデル バージョンを、セマンティックに関連する識別子で一意に識別または参照できる可変文字列です。エイリアス は、登録済みモデルの 1 つの バージョン にのみ割り当てることができます。これは、エイリアス がプログラムで使用される場合に一意の バージョン を参照する必要があるためです。また、エイリアス を使用してモデルの状態 (チャンピオン、候補、プロダクション) をキャプチャすることもできます。

`"best"`、`"latest"`、`"production"`、`"staging"` などの エイリアス を使用して、特別な目的を持つモデル バージョン をマークするのが一般的な方法です。

たとえば、モデルを作成し、`"best"` エイリアス を割り当てるとします。`run.use_model` でその特定のモデルを参照できます。

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## モデル タグ
モデル タグ は、1 つ以上の登録済みモデルに属するキーワードまたはラベルです。

モデル タグ を使用して、登録済みモデルをカテゴリに整理したり、モデルレジストリ の検索バーでそれらのカテゴリを検索したりします。モデル タグ は、登録済みモデルカード の上部に表示されます。それらを使用して、登録済みモデルを ML タスク、所有 チーム、または優先度でグループ化することを選択できます。同じモデル タグ を複数の登録済みモデルに追加して、グループ化を可能にすることができます。

{{% alert %}}
モデル タグ は、グループ化と検出可能性のために登録済みモデルに適用されるラベルであり、[モデル エイリアス]({{< relref path="#model-alias" lang="ja" >}}) とは異なります。モデル エイリアス は、モデル バージョン をプログラムでフェッチするために使用する一意の識別子またはニックネームです。タグ を使用してモデルレジストリ 内のタスクを整理する方法の詳細については、[モデルの整理]({{< relref path="./organize-models.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## モデル artifact
モデル artifact は、ログ に記録された [モデル バージョン]({{< relref path="#model-version" lang="ja" >}}) のコレクションです。モデル バージョンは、モデル artifact にログ された順にモデル artifact に保存されます。

モデル artifact には、1 つ以上のモデル バージョンを含めることができます。モデル バージョンがログ に記録されていない場合、モデル artifact は空になる可能性があります。

たとえば、モデル artifact を作成するとします。モデル トレーニング 中に、チェックポイント 中にモデルを定期的に保存します。各チェックポイント は、独自の [モデル バージョン]({{< relref path="#model-version" lang="ja" >}}) に対応します。モデル トレーニング 中に作成されたすべてのモデル バージョンとチェックポイント の保存は、トレーニング スクリプト の最初に作成した同じモデル artifact に保存されます。

次の図は、v0、v1、および v2 の 3 つのモデル バージョンを含むモデル artifact を示しています。

{{< img src="/images/models/mr1c.png" alt="" >}}

[モデル artifact の例はこちら](https://wandb.ai/timssweeney/model_management_docs_official_v0/artifacts/model/mnist-zws7gt0n) をご覧ください。

## 登録済みモデル
登録済みモデルは、モデル バージョン へのポインター (リンク) のコレクションです。登録済みモデルは、同じ ML タスク の候補モデルの「ブックマーク」のフォルダーと考えることができます。登録済みモデルの各「ブックマーク」は、[モデル artifact]({{< relref path="#model-artifact" lang="ja" >}}) に属する [モデル バージョン]({{< relref path="#model-version" lang="ja" >}}) へのポインターです。[モデル タグ]({{< relref path="#model-tags" lang="ja" >}}) を使用して、登録済みモデルをグループ化できます。

登録済みモデルは、多くの場合、単一のモデリング ユースケース またはタスク の候補モデルを表します。たとえば、使用するモデルに基づいて、さまざまな画像分類 タスク の登録済みモデルを作成できます: `ImageClassifier-ResNet50`、`ImageClassifier-VGG16`、`DogBreedClassifier-MobileNetV2` など。モデル バージョンには、登録済みモデルにリンクされた順にバージョン番号が割り当てられます。

[登録済みモデル の例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions) をご覧ください。
