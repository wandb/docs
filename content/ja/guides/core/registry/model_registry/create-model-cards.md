---
title: 機械学習モデルをドキュメント化する
description: モデルカードに説明を追加して、あなたのモデルをドキュメント化しましょう
menu:
  default:
    identifier: create-model-cards
    parent: model-registry
weight: 8
---

モデルレジストリで登録したモデルのモデルカードに説明を追加することで、あなたの機械学習モデルの各種側面をドキュメント化できます。記載する価値のあるテーマ例は以下の通りです。

* **Summary**: モデルが何か、その目的、使用している機械学習フレームワークなどの概要。
* **Training data**: 使用したトレーニングデータ、そのデータセットに対して行われたプロセッシング、そのデータがどこに保存されているか、などについて記述します。
* **Architecture**: モデルのアーキテクチャー、レイヤー構成、特有の設計判断についての情報。
* **Deserialize the model**: チームメンバーがどのようにモデルをメモリにロードできるか、その手順について書きます。
* **Task**: この機械学習モデルがどのようなタスクや課題を解決するために設計されているのか。モデルの想定される能力を分類するものです。
* **License**: モデルの利用に関する法的な条件や許可。モデルユーザーがどのような法的枠組みでモデルを活用できるかを理解する助けとなります。
* **References**: 関連する研究論文、データセット、外部リソースへの引用や参照情報。
* **Deployment**: モデルがどこで、どのようにデプロイされているかの詳細、他のエンタープライズシステムやワークフローオーケストレーションプラットフォームへどのように統合されているかのガイダンス。

## モデルカードに説明を追加する

1. [W&B Model Registry アプリ](https://wandb.ai/registry/model)にアクセスします。
2. モデルカードを作成したい Registered Model の横にある **View details** を選択します。
3. **Model card** セクションに移動します。
{{< img src="/images/models/model_card_example.png" alt="モデルカード例" >}}
4. **Description** 欄に、あなたの機械学習モデルに関する情報を入力します。モデルカードのテキストは [Markdownマークアップ言語](https://www.markdownguide.org/) で整形できます。

例えば、下記の画像は **Credit-card Default Prediction** Registered Model のモデルカードの例です。
{{< img src="/images/models/model_card_credit_example.png" alt="モデルカード クレジットスコアリング" >}}