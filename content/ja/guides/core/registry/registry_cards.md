---
title: コレクションにアノテーションを付ける
menu:
  default:
    identifier: ja-guides-core-registry-registry_cards
    parent: registry
weight: 8
---

コレクションの目的や含まれる Artifacts をわかりやすく伝えるために、読みやすい説明文をコレクションに追加しましょう。

コレクションによっては、トレーニングデータ、モデルのアーキテクチャー、タスク、ライセンス、参考文献、デプロイ方法に関する情報を含めるとよい場合があります。以下に、コレクションに記載しておくと有用なトピックの例を示します。

W&B は、最低限以下の詳細を含めることを推奨します。
* **概要**: コレクションの目的。実験で使用した機械学習フレームワーク。
* **ライセンス**: 機械学習モデルの使用に関連する法的条件および許可。モデルの利用者が、モデルを利用できる法的枠組みを理解するのに役立ちます。一般的なライセンスには、Apache 2.0、MIT、GPL などがあります。
* **参照**: 関連する研究論文、データセット、または外部リソースへの引用や参照。

コレクションにトレーニングデータが含まれる場合は、以下の詳細を追加することを検討してください。
* **トレーニングデータ**: 使用したトレーニングデータの説明。
* **前処理**: トレーニングデータに施した前処理。
* **データの保存**: データの保存場所とアクセス方法。

コレクションに機械学習モデルが含まれる場合は、以下の詳細を追加することを検討してください。
* **アーキテクチャー**: モデルのアーキテクチャー、レイヤー、および特筆すべき設計上の選択に関する情報。
* **タスク**: コレクションのモデルが実行するように設計されている具体的なタスクや問題の種類。モデルの想定する能力の分類です。
* **モデルのデシリアライズ**: チームの誰もがモデルをメモリにロードできるようにするための情報。
* **デプロイメント**: モデルをどのように、どこにデプロイするかの詳細と、ワークフローオーケストレーションプラットフォームなどの他のエンタープライズシステムにモデルを統合する方法に関するガイダンス。

## コレクションに説明を追加する

W&B Registry UI または Python SDK を使用して、対話的またはプログラムでコレクションに説明を追加します。

{{< tabpane text=true >}}
  {{% tab header="W&B Registry UI" %}}
1. [W&B Registry App](https://wandb.ai/registry/) に移動します。
2. コレクションをクリックします。
3. コレクション名の横にある **View details** を選択します。
4. **Description** フィールドにコレクションの情報を入力します。テキストは [Markdown ガイド](https://www.markdownguide.org/) に沿って書式化できます。

  {{% /tab %}}
  {{% tab header="Python SDK" %}}

コレクションの `description` にアクセスするには、[`wandb.Api().artifact_collection()`]({{< relref path="/ref/python/public-api/api.md#artifact_collection" lang="ja" >}}) メソッドを使用します。返されたオブジェクトの `description` プロパティを使って、コレクションの説明を追加または更新します。

`type_name` パラメータにはコレクションのタイプを、`name` パラメータにはコレクションのフルネームを指定します。コレクション名は、プレフィックス「wandb-registry」、レジストリ名、コレクション名をスラッシュで区切ったもので構成されます。

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

以下のコードスニペットを Python スクリプトやノートブックにコピー＆ペーストし、山括弧 (`<>`) で囲まれた値をご自身のものに置き換えてください。

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

例えば、以下の画像は、モデルのアーキテクチャー、想定される使用方法、パフォーマンス情報などを記載したコレクションを示しています。

{{< img src="/images/registry/registry_card.png" alt="コレクション カード" >}}