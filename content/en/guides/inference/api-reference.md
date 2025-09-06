---
title: "API Reference"
linkTitle: "API Reference"
weight: 40
description: >
  Complete API reference for W&B Inference service
---

Learn how to use the W&B Inference API to access foundation models programmatically.

## Endpoint

Access the Inference service at:

```plaintext
https://api.inference.wandb.ai/v1
```

{{< alert title="Important" >}}
To use this endpoint, you need:
- A W&B account with Inference credits
- A valid W&B API key
- A W&B entity (team) and project

In code samples, these appear as `<your-team>/<your-project>`.
{{< /alert >}}

## Available methods

The Inference API supports these methods:

### Chat completions

Create a chat completion using the `/chat/completions` endpoint. This endpoint follows the OpenAI format for sending messages and receiving responses.

To create a chat completion, provide:
- The Inference service base URL: `https://api.inference.wandb.ai/v1`
- Your W&B API key: `<your-api-key>`
- Your W&B entity and project: `<your-team>/<your-project>`
- A model ID from the [available models]({{< relref "models" >}})

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
    # The custom base URL points to W&B Inference
    base_url='https://api.inference.wandb.ai/v1',

    # Get your API key from https://wandb.ai/authorize
    # Consider setting it in the environment as OPENAI_API_KEY instead for safety
    api_key="<your-api-key>",

    # Team and project are required for usage tracking
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
{{< /tabpane >}}

### List supported models

Get all available models and their IDs. Use this to select models dynamically or check what's available.

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

## Response format

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

## API errors

The following table lists common API errors you might encounter:

| Error Code | Message | Cause | Solution |
| ---------- | ------- | ----- | -------- |
| 401 | Authentication failed | Your authentication credentials are incorrect or your W&B project entity and/or name are incorrect. | Ensure you're using the correct API key and that your W&B project name and entity are correct. |
| 403 | Country, region, or territory not supported | Accessing the API from an unsupported location. | Please see [Geographic restrictions]({{< relref "usage-limits#geographic-restrictions" >}}) |
| 429 | Concurrency limit reached for requests | Too many concurrent requests. | Reduce the number of concurrent requests or increase your limits. For more information, see [Usage information and limits]({{< relref "usage-limits" >}}). |
| 429 | You exceeded your current quota, please check your plan and billing details | Out of credits or reached monthly spending cap. | Get more credits or increase your limits. For more information, see [Usage information and limits]({{< relref "usage-limits" >}}). |
| 429 | W&B Inference isn't available for personal accounts. Please switch to a non-personal account to access W&B Inference | The user is on a personal account, which doesn't have access to W&B Inference. | Switch to a non-personal account. If one isn't available, create a Team to create a non-personal account. For more information, see [Personal entities unsupported]({{< relref "usage-limits#personal-entities-unsupported" >}}). |
| 500 | The server had an error while processing your request | Internal server error. | Retry after a brief wait and contact support if it persists. |
| 503 | The engine is currently overloaded, please try again later | Server is experiencing high traffic. | Retry your request after a short delay. |

## Next steps

- Try the [usage examples]({{< relref "examples" >}}) to see the API in action
- Explore models in the [UI]({{< relref "ui-guide" >}}) 