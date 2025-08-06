---
title: APIリファレンス
description: W&B Inference サービスの完全な API リファレンス
linkTitle: API Reference
weight: 40
---

W&B Inference API を使って、ファウンデーションモデルにプログラム経由でアクセスする方法をご紹介します。

{{< alert title="Tip" >}}
API エラーでお困りですか？解決策は [W&B Inference サポート記事](/support/inference/) をご覧ください。
{{< /alert >}}

## エンドポイント

Inference サービスへのアクセスは以下から行います：

```plaintext
https://api.inference.wandb.ai/v1
```

{{< alert title="Important" >}}
このエンドポイントを利用するには、以下が必要です：
- Inference クレジットがある W&B アカウント
- 有効な W&B APIキー
- W&B entity（チーム）と project

コード例中では `<your-team>/<your-project>` の形式で記載しています。
{{< /alert >}}

## 利用可能なメソッド

Inference API では、以下のメソッドをサポートしています：

### Chat completions

`/chat/completions` エンドポイントを使ってチャット補完を作成します。このエンドポイントは OpenAI のフォーマットでメッセージ送受信を行います。

チャット補完を作成するには以下を指定してください：
- Inference サービスのベースURL: `https://api.inference.wandb.ai/v1`
- ご自身の W&B APIキー: `<your-api-key>`
- ご自身の W&B entity および project: `<your-team>/<your-project>`
- [利用可能なモデル]({{< relref "models" >}}) から選んだモデル ID

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
    # カスタムのベース URL を W&B Inference に設定します
    base_url='https://api.inference.wandb.ai/v1',

    # APIキーは https://wandb.ai/authorize から取得できます
    # セキュリティのため、環境変数 OPENAI_API_KEY で管理することをおすすめします
    api_key="<your-api-key>",

    # 利用状況のトラッキングのため、チームとプロジェクト名が必要です
    project="<your-team>/<your-project>",
)

# <model-id> は利用可能なモデルリストから任意のモデル ID に置き換えてください
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

利用可能なモデルとその ID をすべて取得します。モデルを動的に選択したり、現在利用可能なものを確認したい場合に役立ちます。

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

API のレスポンスは OpenAI 互換のフォーマットで返されます：

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

- [使用例]({{< relref "examples" >}}) を試して API の動作を確認しましょう
- [UI]({{< relref "ui-guide" >}}) でモデル一覧を探すこともできます