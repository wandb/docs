---
title: 機械学習モデルをドキュメント化する
description: モデルカードに説明を追加して、モデルを記録します
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-create-model-cards
    parent: model-registry
weight: 8
---

登録済み Models のモデルカードに説明を追加して、機械学習モデルの側面を文書化します。文書化する価値のあるトピックには、以下のようなものがあります。
* **Summary**: モデルの概要、目的、使用している機械学習フレームワークなど。
* **Training data**: 使用したトレーニングデータ、トレーニングデータセットに対して行った前処理、そのデータの保存場所などを説明します。
* **Architecture**: モデルのアーキテクチャー、レイヤー、および特定の設計上の選択に関する情報。
* **Deserialize the model**: チームの誰かがモデルをメモリにロードする方法に関する情報を提供します。
* **Task**: 機械学習モデルが実行するように設計された具体的なタスクや問題の種類。モデルが意図する機能のカテゴリを示します。
* **License**: 機械学習モデルの使用に関連する法的条件と許諾。モデルの利用者が従うべき法的枠組みを理解する助けになります。
* **References**: 関連する研究論文、Datasets、または外部リソースへの引用や参照。
* **Deployment**: モデルをどのように、どこにデプロイするかの詳細、およびワークフローオーケストレーションプラットフォームなどの他のエンタープライズシステムにどのように統合するかに関するガイダンス。

## モデルカードに説明を追加する

1. [W&B モデルレジストリ アプリ](https://wandb.ai/registry/model)に移動します。
2. モデルカードを作成したい登録済み Models の名前の横にある「**View details**」を選択します。
2. 「**Model card**」セクションに移動します。
{{< img src="/images/models/model_card_example.png" alt="モデルカードの例" >}}
3. 「**Description**」フィールドに、機械学習モデルに関する情報を記載します。モデルカード内のテキストは、[Markdown マークアップ言語](https://www.markdownguide.org/)でフォーマットします。

たとえば、以下の画像は、**Credit-card Default Prediction** 登録済み Models のモデルカードを示しています。
{{< img src="/images/models/model_card_credit_example.png" alt="モデルカードのクレジットスコアリング" >}}