---
title: W&B Inference
description: W&B Weave と OpenAI 互換 API を通じて、オープンソースの foundation models にアクセス
weight: 8
---

W&B Inference は、W&B Weave および OpenAI 互換 API を通じて、最先端のオープンソース基盤モデルへのアクセスを提供します。以下のことが可能です。

- ホスティングプロバイダーへの登録や自己ホスト型モデルの導入なしで、AI アプリケーションやエージェントを構築できます
- W&B Weave Playground で [対応モデル]({{< relref "models" >}}) を試すことができます

Weave を使うことで、W&B Inference で動作するアプリケーションのトレース、評価、モニタリング、改善が可能です。

## クイックスタート

Python を使ったシンプルな例をご紹介します。

```python
import openai

client = openai.OpenAI(
    # カスタム base URL は W&B Inference を指します
    base_url='https://api.inference.wandb.ai/v1',
    
    # https://wandb.ai/authorize から自分の APIキー を取得してください
    api_key="<your-api-key>",
    
    # Team と project は利用状況のトラッキングに必須です
    project="<your-team>/<your-project>",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
)

print(response.choices[0].message.content)
```

## 次のステップ

1. [利用可能なモデル]({{< relref "models" >}}) と [利用情報および制限]({{< relref "usage-limits" >}}) を確認しましょう
2. [事前準備]({{< relref "prerequisites" >}}) を使ってアカウントをセットアップしましょう
3. [API]({{< relref "api-reference" >}}) や [UI]({{< relref "ui-guide" >}}) を使ってサービスを利用しましょう
4. [使用例]({{< relref "examples" >}}) を試してみましょう

## 利用に関する詳細

{{< alert title="重要" color="warning" >}}
W&B Inference クレジットは、Free、Pro、Academic プランに期間限定で付与されます。Enterprise アカウントでは利用可能状況が異なる場合があります。クレジットを使い切ると：

- **Freeユーザー** は Inference を継続利用するために有料プランへのアップグレードが必要です。  
  👉 [Pro または Enterprise へアップグレード](https://wandb.ai/subscriptions)
- **Proユーザー** は、無料クレジット超過分の利用について月ごとに課金されます（デフォルト上限は $6,000/月）。[アカウント階層とデフォルト利用上限]({{< relref "usage-limits#account-tiers-and-default-usage-caps" >}}) をご覧ください
- **Enterprise の利用** は年間 $700,000 に制限されています。請求や上限の引き上げについては担当のアカウントエグゼクティブが対応します。[アカウント階層とデフォルト利用上限]({{< relref "usage-limits#account-tiers-and-default-usage-caps" >}}) をご覧ください

詳細は [価格ページ](https://wandb.ai/site/pricing/) または [モデルごとの料金](https://wandb.ai/site/pricing/inference) をご覧ください。
{{< /alert >}}