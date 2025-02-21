---
title: Annotate collections
menu:
  default:
    identifier: ja-guides-models-registry-registry_cards
    parent: registry
weight: 8
---

コレクションにわかりやすいテキストを追加して、 ユーザー がコレクションの目的とそれに含まれる Artifacts を理解できるようにします。

コレクションによっては、トレーニングデータ、モデルアーキテクチャ、タスク、ライセンス、参考文献、 デプロイメント に関する情報を含めることができます。以下に、コレクションに記述する価値のあるトピックをいくつか示します。

W&B では、少なくとも以下の詳細を含めることを推奨します。
* **概要**: コレクションの目的。 機械学習 実験に使用される 機械学習 フレームワーク。
* **ライセンス**: 機械学習 モデル の使用に関連する法的条件と許可。モデル ユーザー がモデルを利用できる法的 枠組み を理解するのに役立ちます。一般的なライセンスには、Apache 2.0、MIT、GPL があります。
* **参考文献**: 関連する 研究 論文、 データセット 、または外部リソースへの引用または参照。

コレクションにトレーニングデータが含まれている場合は、次の詳細を含めることを検討してください。
* **トレーニングデータ**: 使用されるトレーニングデータについて説明します
* **プロセッシング**: トレーニングデータセットに対して行われた プロセッシング 。
* **データストレージ**: そのデータはどこに保存され、どのように アクセス するか。

コレクションに 機械学習 モデルが含まれている場合は、次の詳細を含めることを検討してください。
* **アーキテクチャー**: モデルアーキテクチャ、レイヤー、および特定の設計上の選択に関する情報。
* **タスク**: コレクションモデルが実行するように設計されているタスクまたは問題の特定のタイプ。モデルの意図された機能を分類したものです。
* **モデルのデシリアライズ**: チームの誰かがモデルをメモリにロードする方法に関する情報を提供します。
* **タスク**: 機械学習 モデルが実行するように設計されているタスクまたは問題の特定のタイプ。モデルの意図された機能を分類したものです。
* **デプロイメント**: モデルの デプロイ 方法と場所の詳細、およびワークフロー オーケストレーション プラットフォーム など、モデルを他のエンタープライズシステムに統合する方法に関するガイダンス。

## コレクションに説明を追加する

W&B Registry UI または Python SDK を使用して、コレクションに説明をインタラクティブまたはプログラムで追加します。

{{< tabpane text=true >}}
  {{% tab header="W&B Registry UI" %}}
1. [https://wandb.ai/registry/](https://wandb.ai/registry/) の W&B Registry に移動します。
2. コレクションをクリックします。
3. コレクション名の横にある [**詳細を表示**] を選択します。
4. [**説明**] フィールドに、コレクションに関する情報を入力します。 [Markdown マークアップ言語](https://www.markdownguide.org/) を使用してテキストの書式を設定します。

  {{% /tab %}}
  {{% tab header="Python SDK" %}}

[`wandb.Api().artifact_collection()`]({{< relref path="/ref/python/public-api/api.md#artifact_collection" lang="ja" >}}) メソッドを使用して、コレクションの説明に アクセス します。返された オブジェクト の `description` プロパティを使用して、コレクションに説明を追加または更新します。

`type_name` パラメータにコレクションのタイプを指定し、`name` パラメータにコレクションのフルネームを指定します。コレクションの名前は、プレフィックス "wandb-registry"、Registry の名前、およびコレクションの名前をフォワードスラッシュで区切ったもので構成されます。

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

次の コードスニペット を Python スクリプト または ノートブック にコピーして貼り付けます。山かっこ (`<>`) で囲まれた 値 を独自の値に置き換えます。

```python
import wandb

api = wandb.Api()

collection = api.artifact_collection(
  type_name = "<collection_type>",
  name = "<collection_name>"
  )


collection.description = "This is a description."
collection.save()
```
  {{% /tab %}}
{{< /tabpane >}}

たとえば、次の画像は、モデルのアーキテクチャ、意図された用途、パフォーマンス情報などを記述したコレクションを示しています。

{{< img src="/images/registry/registry_card.png" alt="Collection card with information about the model architecture, intended use, performance information and more." >}}
