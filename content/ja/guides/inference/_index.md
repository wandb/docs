---
title: W&B Inference
description: W&B Weave と OpenAI 互換 API を使って、オープンソースの foundation models に アクセス できます
menu:
  default:
    identifier: ja-guides-inference-_index
weight: 8
---

W&B Inference を使うと、W&B Weave や OpenAI 互換 API を通じて、主要なオープンソースのファウンデーションモデルにアクセスできます。できること：

- ホスティングプロバイダへの登録やモデルのセルフホストなしで、AI アプリケーションやエージェントを開発できます
- W&B Weave Playground で [利用可能なモデル]({{< relref path="models" lang="ja" >}}) を試せます

Weave を使えば、W&B Inference を活用したアプリケーションのトレース・評価・モニタリング・改善ができます。

## クイックスタート

Python を使ったシンプルな例です：

```python
import openai

client = openai.OpenAI(
    # カスタムの base URL は W&B Inference を指します
    base_url='https://api.inference.wandb.ai/v1',
    
    # APIキーは https://wandb.ai/authorize から取得してください
    api_key="<your-api-key>"
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

1. [利用可能なモデル]({{< relref path="models" lang="ja" >}}) や [利用情報と制限]({{< relref path="usage-limits" lang="ja" >}}) を確認しましょう
2. [前提条件]({{< relref path="prerequisites" lang="ja" >}}) を参考にアカウントをセットアップしましょう
3. [API]({{< relref path="api-reference" lang="ja" >}}) または [UI]({{< relref path="ui-guide" lang="ja" >}}) 経由でサービスを使い始めましょう
4. [使用例]({{< relref path="examples" lang="ja" >}}) を試してみましょう

## 利用に関する詳細

{{< alert title="重要" color="warning" >}}
W&B Inference のクレジットは、Free、Pro、Academic プランに期間限定で付与されています。Enterprise アカウントの場合は内容が異なる場合があります。クレジットがなくなると：

- **Free ユーザー** は利用を続けるために有料プランへのアップグレードが必要です。  
  👉 [Pro または Enterprise へのアップグレード](https://wandb.ai/subscriptions)
- **Pro ユーザー** は、無料クレジットを超えた利用分について毎月課金されます（デフォルト上限は $6,000/月）。[アカウント階層と利用上限]({{< relref path="usage-limits#account-tiers-and-default-usage-caps" lang="ja" >}}) を参照してください
- **Enterprise 利用** は年間 $700,000 までが上限です。課金や上限の引き上げは担当営業がご対応します。[アカウント階層と利用上限]({{< relref path="usage-limits#account-tiers-and-default-usage-caps" lang="ja" >}}) をご覧ください

詳しくは [料金ページ](https://wandb.ai/site/pricing/) または [モデルごとのコスト](https://wandb.ai/site/pricing/inference) を参照してください。
{{< /alert >}}