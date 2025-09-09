---
title: モデルレジストリの用語と概念
description: Model Registry の用語と概念
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-model-management-concepts
    parent: model-registry
weight: 2
---

W&B Model Registry を構成する主な要素は次のとおりです: [モデル バージョン]({{< relref path="#model-version" lang="ja" >}})、[モデル アーティファクト]({{< relref path="#model-artifact" lang="ja" >}})、[登録済みモデル]({{< relref path="#registered-model" lang="ja" >}})。

## モデル バージョン
モデル バージョンは、単一のモデル チェックポイントを表します。実験内のモデルとそのファイルを、ある時点のスナップショットとして切り出したものです。

モデル バージョンは、学習済みモデルを記述するデータとメタデータを収めた不変のディレクトリです。W&B では、後からモデルのアーキテクチャや学習済みパラメータを保存・復元できるよう、モデル バージョンにファイルを追加することを推奨します。

モデル バージョンは 1 つだけの [モデル アーティファクト]({{< relref path="#model-artifact" lang="ja" >}}) に属します。ゼロ個以上の[登録済みモデル]({{< relref path="#registered-model" lang="ja" >}}) に関連付けることができます。モデル バージョンは、ログした順にモデル アーティファクトへ保存されます。W&B は、同じモデル アーティファクトにログするモデルの内容が直前のモデル バージョンと異なると検出した場合、自動で新しいモデル バージョンを作成します。

モデル バージョンには、使用しているモデリング ライブラリのシリアライズ手順で生成されるファイル（例: [PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html)、[Keras](https://www.tensorflow.org/guide/keras/save_and_serialize)）を保存します。

## モデル エイリアス
モデル エイリアスは、登録済みモデル内のモデル バージョンを意味的に表す識別子として一意に参照できる、変更可能な文字列です。エイリアスは同じ登録済みモデルの中で 1 つのバージョンにしか割り当てられません。プログラムから使用する際、エイリアスは特定のバージョンを一意に指す必要があるためです。また、エイリアスでモデルの状態（チャンピオン、候補、プロダクション など）を表すこともできます。

"best"、"latest"、"production"、"staging" といったエイリアスで、特別な目的のモデル バージョンに印を付けるのが一般的です。

たとえば、モデルを作成して "best" エイリアスを付けたとします。`run.use_model` を使ってそのモデルを参照できます。

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## モデル タグ
モデル タグは、1 つ以上の登録済みモデルに付与するキーワードまたはラベルです。

モデル タグを使って登録済みモデルをカテゴリ分けし、Model Registry の検索バーでそのカテゴリを検索できます。モデル タグは登録済みモデルのカード上部に表示されます。これを使って、ML タスク、所有チーム、優先度などで登録済みモデルをグループ化できます。同じモデル タグを複数の登録済みモデルに付けて、グループ化を可能にします。

{{% alert %}}
グループ化と検出可能性のために登録済みモデルに適用するラベルであるモデル タグは、[モデル エイリアス]({{< relref path="#model-alias" lang="ja" >}}) とは異なります。モデル エイリアスは、プログラムでモデル バージョンを取得するために使う一意の識別子（ニックネーム）です。Model Registry のタスクを整理するためのタグの使い方は、[モデルを整理する]({{< relref path="./organize-models.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## モデル アーティファクト
モデル アーティファクトは、ログに記録された[モデル バージョン]({{< relref path="#model-version" lang="ja" >}})の集合です。モデル バージョンは、ログした順にモデル アーティファクトへ保存されます。

モデル アーティファクトには 1 つ以上のモデル バージョンを含められます。モデル バージョンがまだログされていない場合、モデル アーティファクトは空のことがあります。

たとえば、モデル アーティファクトを作成し、トレーニング中にチェックポイントで定期的にモデルを保存する場合、各チェックポイントはそれぞれの[モデル バージョン]({{< relref path="#model-version" lang="ja" >}})に対応します。トレーニングとチェックポイント保存の間に作られたすべてのモデル バージョンは、トレーニング スクリプトの開始時に作成した同じモデル アーティファクトに保存されます。

次の画像は、v0、v1、v2 の 3 つのモデル バージョンを含むモデル アーティファクトを示しています。

{{< img src="/images/models/mr1c.png" alt="モデルレジストリの概念" >}}

[モデル アーティファクトの例はこちらをご覧ください](https://wandb.ai/timssweeney/model_management_docs_official_v0/artifacts/model/mnist-zws7gt0n)。

## 登録済みモデル
登録済みモデルは、モデル バージョンへのポインター（リンク）の集合です。これは、同じ ML タスクに対する候補モデルの「ブックマーク」をまとめたフォルダーと考えられます。登録済みモデルの各「ブックマーク」は、[モデル アーティファクト]({{< relref path="#model-artifact" lang="ja" >}}) に属する[モデル バージョン]({{< relref path="#model-version" lang="ja" >}})へのポインターです。[モデル タグ]({{< relref path="#model-tags" lang="ja" >}})を使って登録済みモデルをグループ化できます。

登録済みモデルは、多くの場合、単一のモデリング ユースケースやタスクに対する候補モデルを表します。たとえば、使用するモデルに基づいて、さまざまな画像分類タスク向けの登録済みモデルを作成できます: `ImageClassifier-ResNet50`、`ImageClassifier-VGG16`、`DogBreedClassifier-MobileNetV2` など。モデル バージョンには、登録済みモデルにリンクされた順にバージョン番号が割り当てられます。

[登録済みモデルの例はこちらをご覧ください](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions)。