---
title: "API Reference"
linkTitle: "API Reference"
weight: 40
description: >
  Complete API reference for W&B Inference service
---

Learn how to use the W&B Inference API to access foundation models programmatically.

{{< alert title="Tip" >}}
Having trouble with API errors? See [W&B Inference support articles](/support/inference/) for solutions.
{{< /alert >}}

## Endpoint

Access the Inference service at:

```plaintext
https://api.inference.wandb.ai/v1
```

{{< alert title="Important" >}}
To use this endpoint, you need:
- A W&B account with Inference credits
- A valid W&B API key

If you belong to more than one team or want to attribute your usage to a project you will also need team and project IDs. In code samples, these appear as `<your-team>/<your-project>`. Your default entity and the project name `inference` will be used if unspecified.
{{< /alert >}}

## Available methods

The Inference API supports these methods:

### Chat completions

Create a chat completion using the `/chat/completions` endpoint. This endpoint follows the OpenAI format for sending messages and receiving responses.

To create a chat completion, provide:
- The Inference service base URL: `https://api.inference.wandb.ai/v1`
- Your W&B API key: `<your-api-key>`
- Optional: Your W&B team and project: `<your-team>/<your-project>`
- A model ID from the [available models]({{< relref "models" >}})

{{< tabpane text=true >}}
{{% tab header="Python" value="python" %}}

```python
import openai

client = openai.OpenAI(
    # The custom base URL points to W&B Inference
    base_url='https://api.inference.wandb.ai/v1',

    # Get your API key from https://wandb.ai/authorize
    # Consider setting it in the environment as OPENAI_API_KEY instead for safety
    api_key="<your-api-key>",

    # Optional: Team and project for usage tracking
    project="<your-team>/<your-project>",
)

# Replace <model-id> with any model ID from the available models list
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
{{< /tabpane >}}

#### Response format

The API returns responses in OpenAI-compatible format:

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

### List supported models

Get all available models and their IDs. Use this to select models dynamically or check what's available.

{{< tabpane text=true >}}
{{% tab header="Python" value="python" %}}

```python
import openai

client = openai.OpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key="<your-api-key>",
    project="<your-team>/<your-project>"  # Optional, for usage tracking
)

response = client.models.list()

for model in response.data:
    print(model.id)
```

{{% /tab %}}
{{% tab header="Bash" value="bash" %}}

```bash
curl https://api.inference.wandb.ai/v1/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "OpenAI-Project: <your-team>/<your-project>"
```

{{% /tab %}}
{{< /tabpane >}}

#### Response format

The API returns responses in OpenAI-compatible format:

```json
{
  "object": "list",
  "data": [
    {
      "id": "deepseek-ai/DeepSeek-V3.1",
      "object": "model",
      "created": 0,
      "owned_by": "system",
      "root": "deepseek-ai/DeepSeek-V3.1"
    },
    {
      "id": "openai/gpt-oss-20b",
      "object": "model",
      "created": 0,
      "owned_by": "system",
      "root": "openai/gpt-oss-20b"
    },
    ...
  ]
}
```

## Next steps

- Try the [usage examples]({{< relref "examples" >}}) to see the API in action
- Explore models in the [UI]({{< relref "ui-guide" >}})
