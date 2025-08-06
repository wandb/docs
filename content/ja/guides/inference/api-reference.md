---
title: API リファレンス
description: W&B Inference サービスの完全な API リファレンス
linkTitle: API Reference
menu:
  default:
    identifier: ja-guides-inference-api-reference
weight: 40
---

W&B Inference API を使って、ファウンデーションモデルにプログラムでアクセスする方法を紹介します。

{{< alert title="ヒント" >}}
API エラーでお困りですか？解決策は [W&B Inference サポート記事](/support/inference/) をご覧ください。
{{< /alert >}}

## エンドポイント

Inference サービスへのアクセス先：

```plaintext
https://api.inference.wandb.ai/v1
```

{{< alert title="重要" >}}
このエンドポイントを利用するには、以下が必要です：
- Inference クレジット付きの W&B アカウント
- 有効な W&B APIキー
- W&B の entity（チーム）と project

コード例では、`<your-team>/<your-project>` の形式で表現されています。
{{< /alert >}}

## 利用可能なメソッド

Inference API では次のメソッドが利用できます：

### Chat completions

`/chat/completions` エンドポイントを使ってチャット補完を作成できます。このエンドポイントは、OpenAI形式でメッセージ送信と応答取得を行います。

チャット補完を作成する際に必要なもの：
- Inference サービスのベースURL: `https://api.inference.wandb.ai/v1`
- ご自身の W&B APIキー: `<your-api-key>`
- ご自身の W&B entity と project: `<your-team>/<your-project>`
- [利用可能なモデル]({{< relref path="models" lang="ja" >}}) のモデルID

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
    # カスタム base URL は W&B Inference を指します
    base_url='https://api.inference.wandb.ai/v1',

    # APIキーは https://wandb.ai/authorize で取得可能
    # 安全のため、環境変数 OPENAI_API_KEY に設定する方法もおすすめします
    api_key="<your-api-key>"
)

# <model-id> を利用可能なモデルリストから任意のモデルIDに置き換えてください
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

### サポートされているモデルの一覧取得

利用可能なモデルやそのIDを一覧で取得できます。この情報を使って動的にモデルを選択したり、どのモデルが使えるか確認できます。

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
    api_key="<your-api-key>"
)

response = client.models.list()

for model in response.data:
    print(model.id)
```

{{% /tab %}}
{{< /tabpane >}}

## レスポンス形式

この API の応答は OpenAI 互換の形式です：

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

- [使用例]({{< relref path="examples" lang="ja" >}}) を参考に API の動作を試してみましょう
- [UI]({{< relref path="ui-guide" lang="ja" >}}) でモデルを調べてみましょう