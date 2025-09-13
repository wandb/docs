---
title: W&B 推論
description: W&B Weave と OpenAI 互換 API を通じて、オープンソースの基盤モデルにアクセス
menu:
  default:
    identifier: ja-guides-inference-_index
weight: 8
---

W&B Inference を使用すると、W&B Weave と OpenAI 互換 API を介して、主要なオープンソース基盤モデルにアクセスできます。以下が可能です:
- ホスティングプロバイダーへの登録やモデルのセルフホストなしで、AI アプリケーションやエージェントを構築
- W&B Weave Playground で [サポートされているモデル]({{< relref path="models" lang="ja" >}}) を試す
Weave を使用すると、W&B Inference を利用したアプリケーションを追跡、評価、監視、改善できます。
## クイックスタート
Python を使用した簡単な例を次に示します。
```python
import openai

client = openai.OpenAI(
    # カスタムベース URL は W&B Inference を指します
    base_url='https://api.inference.wandb.ai/v1',

    # API キーは https://wandb.ai/authorize で取得してください
    api_key="<your-api-key>",

    # 使用状況の追跡には Team と Project が必須です
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
1. [利用可能な Models]({{< relref path="models" lang="ja" >}}) および [使用情報と制限]({{< relref path="usage-limits" lang="ja" >}}) を確認する
2. [前提条件]({{< relref path="prerequisites" lang="ja" >}}) に従ってアカウントをセットアップする
3. [API]({{< relref path="api-reference" lang="ja" >}}) または [UI]({{< relref path="ui-guide" lang="ja" >}}) を通じてサービスを利用する
4. [使用例]({{< relref path="examples" lang="ja" >}}) を試す
## 使用の詳細
{{< alert title="重要" color="warning" >}}
W&B Inference クレジットは、Free、Pro、および Academic プランに期間限定で付属します。Enterprise アカウントでは利用可能性が異なる場合があります。クレジットがなくなると:
- **Free Users** は、Inference を継続して使用するには有料プランにアップグレードする必要があります。
  [Pro または Enterprise にアップグレード](https://wandb.ai/subscriptions)
- **Pro Users** は、無料クレジットを超える使用量に対して月額で請求され、月額 $6,000 のデフォルト上限までとなります。[アカウントティアとデフォルトの使用上限]({{< relref path="usage-limits#account-tiers-and-default-usage-caps" lang="ja" >}}) を参照してください
- **Enterprise の使用量** は年間 $700,000 に制限されています。担当のアカウント エグゼクティブが請求と上限引き上げを処理します。[アカウントティアとデフォルトの使用上限]({{< relref path="usage-limits#account-tiers-and-default-usage-caps" lang="ja" >}}) を参照してください
詳細については、[料金ページ](https://wandb.ai/site/pricing/) にアクセスするか、[モデル固有の費用](https://wandb.ai/site/pricing/inference) を参照してください。
{{< /alert >}}