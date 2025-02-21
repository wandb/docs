---
title: Document machine learning model
description: モデルカードに説明を追加して、あなたのモデルを文書化しましょう。
menu:
  default:
    identifier: ja-guides-models-registry-model_registry-create-model-cards
    parent: model-registry
weight: 8
---

登録されたモデルのモデルカードに説明を追加して、 機械学習 モデルの側面を文書化します。文書化する価値のあるトピックには、次のものがあります。

* **概要** : モデルの概要。モデルの目的。モデルが使用する 機械学習 フレームワークなど。
* **トレーニングデータ** : 使用したトレーニングデータ、トレーニング データセット で行った処理、データの保存場所などを記述します。
* **アーキテクチャー** : モデルのアーキテクチャー、レイヤー、および特定の設計上の選択に関する情報。
* **モデルのデシリアライズ** : チームの誰かがモデルをメモリーにロードする方法に関する情報を提供します。
* **タスク** : 機械学習 モデルが実行するように設計されている特定のタイプのタスクまたは問題。モデルの意図された機能の分類です。
* **ライセンス** : 機械学習 モデルの使用に関連する法的条件と許可。モデル ユーザーがモデルを利用できる法的枠組みを理解するのに役立ちます。
* **参考文献** : 関連する研究論文、データセット、または外部リソースの引用または参考文献。
* **デプロイメント** : モデルのデプロイ方法と場所の詳細、および ワークフロー オーケストレーション プラットフォーム などの他のエンタープライズ システムにモデルを統合する方法に関するガイダンス。

## モデルカードに説明を追加する

1. W&B Model Registry アプリ ( [https://wandb.ai/registry/model](https://wandb.ai/registry/model) ) に移動します。
2. モデルカードを作成する Registered Model の名前の横にある [**View details**] を選択します。
2. [**Model card**] セクションに移動します。
{{< img src="/images/models/model_card_example.png" alt="" >}}
3. [**Description**] フィールドに、 機械学習 モデルに関する情報を入力します。 [Markdown markup language](https://www.markdownguide.org/) でモデルカード内のテキストをフォーマットします。

たとえば、次の画像は、**Credit-card Default Prediction** Registered Model のモデルカードを示しています。
{{< img src="/images/models/model_card_credit_example.png" alt="" >}}