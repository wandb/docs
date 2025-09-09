---
title: 前提条件
description: W&B Inference を使用するための 環境 をセットアップする
linkTitle: Prerequisites
menu:
  default:
    identifier: ja-guides-inference-prerequisites
weight: 1
---

API または UI を通じて W&B Inference サービスを使用する前に、次の手順を完了してください。

{{< alert title="ヒント" >}}
始める前に、費用と制限を把握するために [使用量と制限]({{< relref path="usage-limits" lang="ja" >}}) を確認してください。
{{< /alert >}}

## W&B アカウントと Project を設定する

W&B Inference に アクセス するには、次が必要です:

1. **W&B アカウント**  
   登録は [W&B](https://app.wandb.ai/login?signup=true) から

2. **W&B APIキー**  
   APIキーは [https://wandb.ai/authorize](https://wandb.ai/authorize) から取得できます

3. **W&B Project**  
   使用状況を追跡するため、W&B アカウントで Project を作成してください

## 環境を設定する (Python)

Python で Inference API を使用するには、次も必要です:

1. 上記の一般的な前提条件を満たす

2. 必要な ライブラリ をインストールする:

   ```bash
   pip install openai weave
   ```

{{< alert title="注意" >}}
`weave` ライブラリは任意ですが、推奨します。LLM アプリケーションをトレースできます。詳細は [Weave クイックスタート]({{< relref path="../quickstart" lang="ja" >}}) を参照してください。

Weave と併用した W&B Inference のコード サンプルは、[使用例]({{< relref path="examples" lang="ja" >}}) を参照してください。
{{< /alert >}}

## 次のステップ

前提条件を満たしたら:

- 利用可能なエンドポイントについては [API リファレンス]({{< relref path="api-reference" lang="ja" >}}) を確認してください
- サービスの動作は [使用例]({{< relref path="examples" lang="ja" >}}) を試して確認してください
- Web インターフェースから Models に アクセス するには、[UI ガイド]({{< relref path="ui-guide" lang="ja" >}}) を参照してください