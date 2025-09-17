---
title: API リファレンス
description: W&B Inference Service の完全な API リファレンス
linkTitle: API Reference
menu:
  default:
    identifier: ja-guides-inference-api-reference
weight: 40
---

W&B Inference API を使用して、プログラムから基盤モデルにアクセスする方法を学びます。
{{< alert title="ヒント" >}}
API エラーでお困りですか？ 解決策については、[W&B Inference サポート記事](/support/inference/)をご覧ください。
{{< /alert >}}

## エンドポイント

Inference サービスには次の URL でアクセスできます。
```plaintext
https://api.inference.wandb.ai/v1
```
{{< alert title="重要" >}}
このエンドポイントを使用するには、以下が必要です。
- Inference クレジットを持つ W&B アカウント
- 有効な W&B API キー
- W&B の Entity と Project

コードサンプルでは、これらは `<your-team>/<your-project>` のように表示されます。
{{< /alert >}}

## 利用可能なメソッド

Inference API は次のメソッドをサポートしています。

### チャット補完

`/chat/completions` エンドポイントを使用して、チャット補完を作成します。このエンドポイントは、OpenAI のメッセージ送信・受信形式に従います。

チャット補完を作成するには、以下を指定します。
- Inference サービスのベース URL: `https://api.inference.wandb.ai/v1`
- W&B API キー: `<your-api-key>`
- W&B の Entity と Project: `<your-team>/<your-project>`
- [利用可能な Models]({{< relref path="models" lang="ja" >}}) からの Model ID

{{< tabpane text=true >}}
{{% tab header="Bash" value="bash" %}}
```bash
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "OpenAI-Project: <your-team>/<your-project>" \
  -d '{
    "model": "<model-id>",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "Tell me a joke." }
    ]
  }'
```
{{% /tab %}}
{{% tab header="Python" value="python" %}}
```python
import openai

client = openai.OpenAI(
    # カスタムベース URL は W&B Inference を指します
    base_url='https://api.inference.wandb.ai/v1',

    # API キーは https://wandb.ai/authorize から取得してください
    # 安全のため、代わりに環境変数 OPENAI_API_KEY として設定することを検討してください
    api_key="<your-api-key>",

    # Entity と Project は使用状況のトラッキングに必要です
    project="<your-team>/<your-project>",
)

# <model-id> を利用可能なモデルリストから任意のモデル ID に置き換えてください
response = client.chat.completions.create(
    model="<model-id>",
    messages=[
        {"role": "system", "content": "<your-system-prompt>"},
        {"role": "user", "content": "<your-prompt>"}
    ],
)

print(response.choices[0].message.content)
```
{{% /tab %}}
{{< /tabpane >}}

### サポートされている Models をリスト表示

利用可能なすべての Models とその ID を取得します。これを使用して、Models を動的に選択したり、利用可能なものを確認したりできます。

{{< tabpane text=true >}}
{{% tab header="Bash" value="bash" %}}
```bash
curl https://api.inference.wandb.ai/v1/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "OpenAI-Project: <your-team>/<your-project>" 
```
{{% /tab %}}
{{% tab header="Python" value="python" %}}
```python
import openai

client = openai.OpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key="<your-api-key>",
    project="<your-team>/<your-project>"
)

response = client.models.list()

for model in response.data:
    print(model.id)
```
{{% /tab %}}
{{< /tabpane >}}

## レスポンス形式

API は OpenAI 互換の形式でレスポンスを返します。
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here's a joke for you..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 50,
    "total_tokens": 75
  }
}
```

## 次のステップ

- [使用例]({{< relref path="examples" lang="ja" >}}) を試して、API の動作を確認します
- [UI]({{< relref path="ui-guide" lang="ja" >}}) で Models を探索します