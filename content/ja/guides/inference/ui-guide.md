---
title: UI ガイド
description: Web インターフェースを通じて W&B Inference Models にアクセス
weight: 60
---

W&B Inference サービスをウェブ UI から利用する方法をご紹介します。UI を使う前に、[事前準備]({{< relref "prerequisites" >}}) を完了してください。

## Inference サービスへのアクセス

Inference サービスには、以下の 3 つの方法でアクセスできます。

### 直接リンク

[https://wandb.ai/inference](https://wandb.ai/inference) にアクセスしてください。

### Inference タブから

1. [https://wandb.ai/](https://wandb.ai/) で自分の W&B アカウントにアクセスします
2. 左サイドバーから **Inference** を選択します
3. 利用可能なモデルやモデルの情報が表示されたページが開きます

{{< img src="/images/inference/inference-playground-single.png" alt="Playground で Inference モデルを利用する様子" >}}

### Playground タブから

1. 左サイドバーから **Playground** を選択します。Playground のチャット UI が開きます
2. LLM のドロップダウンリストで **W&B Inference** にカーソルを合わせます。右側に利用可能なモデルのドロップダウンが表示されます
3. モデルのドロップダウンからは次のことができます:
   - モデル名をクリックして [Playground で試す](#try-a-model-in-the-playground)
   - [複数のモデルを比較](#compare-multiple-models)

{{< img src="/images/inference/inference-playground.png" alt="Playground の Inference モデルドロップダウン" >}}

## Playground でモデルを試す

[モデルを選択](#access-the-inference-service) した後は、それを Playground でテストできます。実行可能な操作は次の通りです。

- [モデルの設定やパラメータをカスタマイズ](https://weave-docs.wandb.ai/guides/tools/playground#customize-settings)
- [メッセージの追加、再試行、編集、削除](https://weave-docs.wandb.ai/guides/tools/playground#message-controls)
- [設定をカスタマイズしたモデルの保存と再利用](https://weave-docs.wandb.ai/guides/tools/playground#saved-models)
- [複数のモデルを比較](#compare-multiple-models)

## 複数のモデルを比較する

Playground では、Inference モデル同士を並べて比較できます。[比較ビュー] には次の 2 つの方法からアクセスできます。

### Inference タブから

1. 左サイドバーから **Inference** を選択します。利用可能なモデルのページが表示されます
2. モデルカード上（モデル名以外の部分）をクリックして選択します。カードの枠が青色に変わります
3. 比較したいすべてのモデルで同じ操作を繰り返します
4. 選択されたどのカードでも **Compare N models in the Playground** をクリックします。`N` には選択したモデル数が表示されます
5. 比較ビューが開きます

これでモデルを比較したり、[Playground でモデルを試す](#try-a-model-in-the-playground) で紹介したすべての機能が使えます。

{{< img src="/images/inference/inference-playground-compare.png" alt="Playground で複数モデルを選択・比較" >}}

### Playground タブから

1. 左サイドバーから **Playground** を選択します。Playground のチャット UI が開きます
2. LLM のドロップダウンリストで **W&B Inference** にカーソルを合わせます。右側にモデルのドロップダウンが表示されます
3. ドロップダウンから **Compare** を選択します。**Inference** タブが表示されます
4. モデルカード上（モデル名以外の部分）をクリックして選択します。カードの枠が青色に変わります
5. 比較したいすべてのモデルで同じ操作を繰り返します
6. 選択されたどのカードでも **Compare N models in the Playground** をクリックします。比較ビューが開きます

これでモデルを比較したり、[Playground でモデルを試す](#try-a-model-in-the-playground) で紹介したすべての機能が利用できます。

## 請求・利用情報の確認

組織管理者は、W&B UI からクレジット残高、利用履歴、今後の請求額の確認ができます。

1. UI 内の W&B **Billing** ページにアクセスします
2. 右下の Inference の請求情報カードを探します
3. ここから次の操作が可能です:
   - **View usage** をクリックして利用状況を確認
   - 今後の Inference 請求予定（有料プラン向け）の確認

{{< alert title="ヒント" >}}
[Inference 料金ページ](https://wandb.ai/site/pricing/inference) でモデルごとの料金詳細を確認できます。
{{< /alert >}}

## 次のステップ

- [利用可能なモデル]({{< relref "models" >}}) を確認して目的に合ったものを探しましょう
- プログラムによるアクセスには [API]({{< relref "api-reference" >}}) をお試しください
- [使用例]({{< relref "examples" >}}) ではコードサンプルを掲載しています