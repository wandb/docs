---
title: 前提条件
description: W&B Inference を使うための環境をセットアップする
linkTitle: Prerequisites
weight: 1
---

W&B Inference サービスを API または UI から利用する前に、以下の手順を完了してください。

{{< alert title="ヒント" >}}
始める前に、[利用情報と制限]({{< relref "usage-limits" >}}) を確認して、コストや制約を理解しましょう。
{{< /alert >}}

## W&B アカウントとプロジェクトのセットアップ

W&B Inference を利用するには、次のものが必要です。

1. **W&B アカウント**  
   [W&B](https://app.wandb.ai/login?signup=true) でサインアップ

2. **W&B APIキー**  
   [https://wandb.ai/authorize](https://wandb.ai/authorize) から APIキー を取得

3. **W&B プロジェクト**  
   利用状況を追跡するために、W&B アカウント内でプロジェクトを作成

## 環境のセットアップ（Python）

Python で Inference API を使う場合、さらに以下が必要です。

1. 上記の一般的な要件をすべて完了

2. 必要なライブラリをインストール:

   ```bash
   pip install openai weave
   ```

{{< alert title="注意" >}}
`weave` ライブラリは任意ですが、おすすめです。これにより、LLM アプリケーションのトレースが可能になります。詳しくは [Weave クイックスタート]({{< relref "../quickstart" >}}) をご覧ください。

W&B Inference と Weave を組み合わせた [利用例]({{< relref "examples" >}}) で、コードサンプルを確認できます。
{{< /alert >}}

## 次のステップ

前提条件を完了したら：

- 利用可能なエンドポイントについては [API リファレンス]({{< relref "api-reference" >}}) をチェック
- サービスの動作は [利用例]({{< relref "examples" >}}) で確認できます
- Webインターフェースでモデルにアクセスするには [UI ガイド]({{< relref "ui-guide" >}}) を利用してください