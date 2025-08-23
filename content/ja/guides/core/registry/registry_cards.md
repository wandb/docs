---
title: コレクションに注釈を付ける
menu:
  default:
    identifier: ja-guides-core-registry-registry_cards
    parent: registry
weight: 8
---

コレクションにわかりやすいテキストを追加して、ユーザーがコレクションの目的や含まれる Artifacts を理解しやすくしましょう。

コレクションによっては、トレーニングデータ、モデルアーキテクチャー、タスク、ライセンス、参考文献、デプロイメントに関する情報を含めるとよいでしょう。以下に、コレクションで記載すべきトピックを挙げます。

W&B では、最低限次の内容を含めることを推奨しています：
* **Summary**: コレクションの目的や、機械学習実験で使用した機械学習フレームワークについて。
* **License**: 機械学習モデルの利用に関する法的条件や許可事項。モデル利用者がどのような法的枠組みでモデルを利用できるか理解するのに役立ちます。一般的なライセンスには Apache 2.0、MIT、GPL などがあります。
* **References**: 関連する論文、データセット、外部リソースへの引用や参照。

コレクションにトレーニングデータが含まれる場合、さらに次の情報も含めると良いでしょう：
* **Training data**: 使用したトレーニングデータの説明
* **Processing**: トレーニングデータセットに対して行ったプロセッシングの内容
* **Data storage**: データがどこに保存されており、どのようにアクセスできるか

コレクションに機械学習モデルが含まれる場合は、以下も考慮してください：
* **Architecture**: モデルのアーキテクチャー、レイヤー構造、特有の設計選択に関する情報
* **Task**: コレクション内モデルが想定しているタスクや問題の種類。そのモデルがどんな能力を持つかの分類です。
* **Deserialize the model**: チームのメンバーがモデルをメモリに読み込む方法について記載
* **Deployment**: モデルがどこに、どのようにデプロイされているか、またワークフローオーケストレーションプラットフォームなどの他の企業システムと統合する方法についてのガイダンス

## コレクションに説明を追加する

W&B Registry UI または Python SDK を使って、インタラクティブまたはプログラム的にコレクションに説明を追加できます。

{{< tabpane text=true >}}
  {{% tab header="W&B Registry UI" %}}
1. [W&B Registry App](https://wandb.ai/registry/) にアクセスします。
2. コレクションをクリックします。
3. コレクション名の横にある **View details** を選択します。
4. **Description** フィールド内で、コレクションに関する情報を入力します。テキストの装飾には [Markdown マークアップ言語](https://www.markdownguide.org/)を利用できます。

  {{% /tab %}}
  {{% tab header="Python SDK" %}}

[`wandb.Api().artifact_collection()`]({{< relref path="/ref/python/public-api/api.md#artifact_collection" lang="ja" >}}) メソッドでコレクションの説明にアクセスできます。返されたオブジェクトの `description` プロパティを使って、コレクションへの説明文の追加や更新が可能です。

`type_name` パラメータにはコレクションのタイプ、`name` にはコレクションのフルネームを指定します。コレクション名は “wandb-registry” というプレフィックス、Registry 名、コレクション名をスラッシュで区切った形です。

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

以下のコードスニペットを Python スクリプトやノートブックにコピー＆ペーストし、`<>`で囲まれた値を自分のものに置き換えてください。

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

例えば、以下の画像はモデルのアーキテクチャー、想定用途、性能情報などをドキュメント化したコレクションの例です。

{{< img src="/images/registry/registry_card.png" alt="Collection card" >}}