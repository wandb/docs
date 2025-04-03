---
title: Annotate collections
menu:
  default:
    identifier: ja-guides-core-registry-registry_cards
    parent: registry
weight: 8
---

コレクションに人に優しいテキストを追加して、ユーザーがコレクションの目的とそれに含まれる Artifacts を理解できるようにします。

コレクションによっては、トレーニングデータ、モデルアーキテクチャー、タスク、ライセンス、参考文献、およびデプロイメントに関する情報を含めることをお勧めします。以下に、コレクションでドキュメント化する価値のあるトピックをいくつか示します。

W&B は、少なくとも以下の詳細を含めることを推奨します。
* **Summary** : コレクションの目的。機械学習実験に使用される機械学習 フレームワーク。
* **License**: 機械学習 モデルの使用に関連する法的条件と許可。これにより、モデルの ユーザー は、モデルを利用できる法的枠組みを理解できます。一般的なライセンスには、Apache 2.0、MIT、GPL などがあります。
* **References**: 関連する 研究 論文、データセット、または外部リソースへの引用または参照。

コレクションにトレーニングデータが含まれている場合は、次の詳細を含めることを検討してください。
* **Training data**: 使用されるトレーニングデータを記述します。
* **Processing**: トレーニングデータセットに対して行われた処理。
* **Data storage**: そのデータはどこに保存され、どのようにアクセスするか。

コレクションに機械学習 モデルが含まれている場合は、次の詳細を含めることを検討してください。
* **Architecture**: モデルアーキテクチャー、レイヤー、および特定の設計の選択に関する情報。
* **Task**: コレクション モデル が実行するように設計されているタスクまたは問題の特定のタイプ。これは、モデルの意図された機能を分類したものです。
* **Deserialize the model**: チームの誰かがモデルをメモリーにロードする方法に関する情報を提供します。
* **Task**: 機械学習 モデル が実行するように設計されているタスクまたは問題の特定のタイプ。これは、モデルの意図された機能を分類したものです。
* **Deployment**: モデルがどのように、どこにデプロイされるかの詳細、およびワークフロー オーケストレーション プラットフォームなどの他のエンタープライズ システムにモデルを統合する方法に関するガイダンス。

## コレクションに説明を追加する

W&B Registry UI または Python SDK を使用して、コレクションにインタラクティブまたはプログラムで説明を追加します。

{{< tabpane text=true >}}
  {{% tab header="W&B Registry UI" %}}
1. [https://wandb.ai/registry/](https://wandb.ai/registry/) で W&B Registry に移動します。
2. コレクションをクリックします。
3. コレクション名の横にある [**詳細を表示**] を選択します。
4. [**Description**] フィールドに、コレクションに関する情報を入力します。[Markdown markup language](https://www.markdownguide.org/) でテキストの書式を設定します。

  {{% /tab %}}
  {{% tab header="Python SDK" %}}

[`wandb.Api().artifact_collection()`]({{< relref path="/ref/python/public-api/api.md#artifact_collection" lang="ja" >}}) メソッドを使用して、コレクションの説明にアクセスします。返されたオブジェクトの `description` プロパティを使用して、コレクションに説明を追加または更新します。

`type_name` パラメータにコレクションのタイプを指定し、`name` パラメータにコレクションのフルネームを指定します。コレクションの名前は、プレフィックス "wandb-registry"、レジストリの名前、およびフォワード スラッシュで区切られたコレクションの名前で構成されます。

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

次のコード スニペットを Python スクリプトまたは ノートブック にコピーして貼り付けます。山かっこ (`<>`) で囲まれた値を独自の値に置き換えます。

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

たとえば、次の画像は、モデルのアーキテクチャー、意図された使用法、パフォーマンス情報などを文書化したコレクションを示しています。

{{< img src="/images/registry/registry_card.png" alt="モデルのアーキテクチャー、意図された使用法、パフォーマンス情報などに関する情報を含むコレクションカード。" >}}
