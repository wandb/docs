---
title: UI ガイド
description: Web インターフェースから W&B Inference の モデル に アクセスする
menu:
  default:
    identifier: ja-guides-inference-ui-guide
weight: 60
---

Web UI から W&B Inference サービスを使う方法を説明します。UI を使い始める前に、[前提条件]({{< relref path="prerequisites" lang="ja" >}}) を完了してください。

## Inference サービスへの アクセス

Inference サービスには、次の 3 つの場所から アクセスできます:

### 直接リンク

[https://wandb.ai/inference](https://wandb.ai/inference) にアクセスします。

### Inference タブから

1. [https://wandb.ai/](https://wandb.ai/) のご自身の W&B アカウントにアクセスします
2. 左サイドバーから **Inference** を選択します
3. 利用可能な モデル と モデル 情報が表示されます

{{< img src="/images/inference/inference-playground-single.png" alt="Playground で Inference モデル を使う" >}}

### Playground タブから

1. 左サイドバーから **Playground** を選択します。Playground のチャット UI が表示されます
2. LLM のドロップダウンリストで **W&B Inference** にカーソルを合わせます。右側に利用可能な モデル のドロップダウンが表示されます
3. モデル のドロップダウンから、次の操作ができます:
   - 任意の モデル 名をクリックして、[Playground で試す](#try-a-model-in-the-playground)
   - [複数の モデル を比較する](#compare-multiple-models)

{{< img src="/images/inference/inference-playground.png" alt="Playground の Inference モデル ドロップダウン" >}}

## Playground で モデル を試す

[モデル を選択](#access-the-inference-service) したら、Playground で試せます。できることは次のとおりです:

- [モデル の 設定 と パラメータ をカスタマイズ](https://weave-docs.wandb.ai/guides/tools/playground#customize-settings)
- [メッセージの追加、再試行、編集、削除](https://weave-docs.wandb.ai/guides/tools/playground#message-controls)
- [カスタム 設定 の モデル を保存して再利用](https://weave-docs.wandb.ai/guides/tools/playground#saved-models)
- [複数の モデル を比較する](#compare-multiple-models)

## 複数の モデル を比較する

Playground で Inference モデル を並べて比較できます。Compare ビューには 2 つの場所から アクセスできます:

### Inference タブから

1. 左サイドバーから **Inference** を選択します。利用可能な モデル のページが表示されます
2. モデル カード上の任意の場所（モデル 名以外）をクリックして選択します。カードの枠が青になります
3. 比較したい モデル ごとに繰り返します
4. 選択したいずれかのカードで **Compare N models in the Playground** をクリックします。`N` は選択した モデル の数です
5. 比較ビューが開きます

これで モデル を比較でき、[Playground で モデル を試す](#try-a-model-in-the-playground) のすべての機能を使えます。

{{< img src="/images/inference/inference-playground-compare.png" alt="Playground で比較する モデル を複数選択" >}}

### Playground タブから

1. 左サイドバーから **Playground** を選択します。Playground のチャット UI が表示されます
2. LLM のドロップダウンリストで **W&B Inference** にカーソルを合わせます。右側に モデル のドロップダウンが表示されます
3. ドロップダウンから **Compare** を選択します。**Inference** タブが表示されます
4. モデル カード上の任意の場所（モデル 名以外）をクリックして選択します。カードの枠が青になります
5. 比較したい モデル ごとに繰り返します
6. 選択したいずれかのカードで **Compare N models in the Playground** をクリックします。比較ビューが開きます

これで モデル を比較でき、[Playground で モデル を試す](#try-a-model-in-the-playground) のすべての機能を使えます。

## 請求と使用状況の 情報 を表示する

組織の管理者は、W&B の UI からクレジット残高、使用履歴、今後の請求を確認できます:

1. UI の W&B **Billing** ページに アクセスします
2. 右下隅にある Inference の請求 情報 カードを探します
3. ここから次のことができます:
   - **View usage** をクリックして、使用状況の推移を確認
   - 今後の Inference の課金額を表示（有料プランの場合）

{{< alert title="ヒント" >}}
モデル ごとの料金の詳細は、[Inference の料金ページ](https://wandb.ai/site/pricing/inference) を参照してください。
{{< /alert >}}

## 次のステップ

- ニーズに合った最適なものを見つけるために、[利用可能な モデル]({{< relref path="models" lang="ja" >}}) を確認
- プログラムからの アクセス には [API]({{< relref path="api-reference" lang="ja" >}}) をお試しください
- コード サンプルは [使用例]({{< relref path="examples" lang="ja" >}}) を参照してください