---
title: 環境のセットアップ
description: W&B Inference を利用するための環境をセットアップする
linkTitle: Prerequisites
menu:
  default:
    identifier: ja-guides-inference-prerequisites
weight: 1
---

W&B Inference サービスを API または UI で利用する前に、以下の手順を完了してください。

{{< alert title="ヒント" >}}
始める前に、[利用情報と制限事項]({{< relref path="usage-limits" lang="ja" >}}) を確認し、コストや条件をご理解ください。
{{< /alert >}}

## W&B アカウントと Project のセットアップ

W&B Inference を利用するには、以下が必要です。

1. **W&B アカウント**  
   [W&B](https://app.wandb.ai/login?signup=true) でサインアップしてください

2. **W&B APIキー**  
   [https://wandb.ai/authorize](https://wandb.ai/authorize) から APIキー を取得してください

3. **W&B Project**  
   利用状況を記録する Project を W&B アカウントで作成してください

## 環境のセットアップ（Python）

Python で Inference API を使う場合は、下記も必要です。

1. 上記の一般的な準備をすべて完了する

2. 必要なライブラリをインストールする:

   ```bash
   pip install openai weave
   ```

{{< alert title="注意" >}}
`weave` ライブラリは任意ですが、推奨されます。LLMアプリケーションのトレースが可能です。詳しくは [Weave クイックスタート]({{< relref path="../quickstart" lang="ja" >}}) をご覧ください。

Weave と W&B Inference の組み合わせ例は [使用例]({{< relref path="examples" lang="ja" >}}) をご参照ください。
{{< /alert >}}

## 次のステップ

準備が完了したら：

- 利用可能なエンドポイントについては [API リファレンス]({{< relref path="api-reference" lang="ja" >}}) をご覧ください
- サービス利用例は [使用例]({{< relref path="examples" lang="ja" >}}) で確認できます
- ウェブインターフェース経由でモデルを利用する方法は [UI ガイド]({{< relref path="ui-guide" lang="ja" >}}) をご覧ください