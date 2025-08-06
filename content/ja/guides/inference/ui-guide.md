---
title: UI ガイド
description: Web インターフェースを通じて W&B Inference models にアクセスする
menu:
  default:
    identifier: ja-guides-inference-ui-guide
weight: 60
---

W&B Inference サービスをウェブ UI から利用する方法を説明します。UI を使う前に、[前提条件]({{< relref path="prerequisites" lang="ja" >}})を完了してください。

## Inference サービスへのアクセス

Inference サービスには、以下の3つの方法でアクセスできます。

### ダイレクトリンク

[https://wandb.ai/inference](https://wandb.ai/inference) へアクセスします。

### Inference タブから

1. [https://wandb.ai/](https://wandb.ai/) で自身の W&B アカウントにアクセス
2. 左サイドバーから **Inference** を選択
3. 利用可能なモデルやモデル情報が表示されます

{{< img src="/images/inference/inference-playground-single.png" alt="Playground で Inference モデルを利用する様子" >}}

### Playground タブから

1. 左サイドバーから **Playground** を選択します。Playground のチャット UI が表示されます
2. LLM ドロップダウンリストで **W&B Inference** にカーソルを合わせます。右側に利用可能なモデルのドロップダウンが表示されます
3. モデルのドロップダウンで以下の操作ができます:
   - 任意のモデル名をクリックして [Playground で試す](#try-a-model-in-the-playground)
   - [複数モデルを比較する](#compare-multiple-models)

{{< img src="/images/inference/inference-playground.png" alt="Playground の Inference モデルドロップダウン" >}}

## Playground でモデルを試す

[モデルを選択した後](#access-the-inference-service)、Playground でモデルを試すことができます。利用可能なアクションは以下の通りです。

- [モデル設定やパラメータのカスタマイズ](https://weave-docs.wandb.ai/guides/tools/playground#customize-settings)
- [メッセージの追加・再試行・編集・削除](https://weave-docs.wandb.ai/guides/tools/playground#message-controls)
- [カスタム設定付きでモデルを保存・再利用](https://weave-docs.wandb.ai/guides/tools/playground#saved-models)
- [複数モデルの比較](#compare-multiple-models)

## 複数モデルの比較

Playground で Inference モデルを並べて比較できます。比較ビューへのアクセス方法は2つあります。

### Inference タブから

1. 左サイドバーで **Inference** を選択します。利用可能なモデル一覧が表示されます
2. 比較したいモデルのカード（モデル名以外の場所）をクリックして選択します。カードの枠が青色に変わります
3. 比較したい他のモデルでも同じ操作を繰り返します
4. 選択したカード上に表示される **Compare N models in the Playground** をクリックします。`N` には選択したモデル数が表示されます
5. 比較ビューが開きます

これでモデルを比較し、[Playground でモデルを試す](#try-a-model-in-the-playground) で説明したすべての機能を利用できます。

{{< img src="/images/inference/inference-playground-compare.png" alt="Playground で複数モデルを選択して比較" >}}

### Playground タブから

1. 左サイドバーで **Playground** を選択します。Playground のチャット UI が表示されます
2. LLM ドロップダウンリストで **W&B Inference** にカーソルを合わせます。モデルのドロップダウンが右側に表示されます
3. ドロップダウンから **Compare** を選択します。**Inference** タブが表示されます
4. 比較したいモデルのカード（モデル名以外の場所）をクリックして選択します。カードの枠が青色に変わります
5. 比較したい他のモデルでも同じ操作を繰り返します
6. 選択したカード上に表示される **Compare N models in the Playground** をクリックします。比較ビューが開きます

これでモデルを比較し、[Playground でモデルを試す](#try-a-model-in-the-playground) で説明したすべての機能を利用できます。

## 課金および利用状況の確認

組織管理者は W&B UI からクレジット残高、利用履歴、今後の請求予定を確認できます。

1. UI の W&B **Billing** ページにアクセス
2. 画面右下の Inference 課金情報カードを見つけます
3. ここから次の操作が可能です:
   - **View usage** をクリックして使用履歴を確認
   - 今後の Inference 請求額を確認（有料プランの場合）

{{< alert title="ヒント" >}}
[Inference の料金ページ](https://wandb.ai/site/pricing/inference) でモデルごとの料金詳細が確認できます。
{{< /alert >}}

## 次のステップ

- [利用可能なモデル一覧]({{< relref path="models" lang="ja" >}}) をチェックし、自分のニーズに合ったモデルを探しましょう
- プログラムからアクセスするには [API]({{< relref path="api-reference" lang="ja" >}}) を試してみましょう
- [利用例]({{< relref path="examples" lang="ja" >}}) でコードサンプルを確認しましょう