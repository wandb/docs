---
description: >-
  How to use JSON mode with W&B Inference.
displayed_sidebar: default
title: "Enable JSON mode"
weight: 3
---


Enabling JSON mode instructs the model to return the response in a valid JSON format. However, the reponse's schema may not be consistent or adhere to a particular structure. For consistent structured JSON responses, we recommend using [ structured output](../structured-output) when possible.

To enable JSON mode, specify it as the "response_format" in the request:

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python {hl_lines=[15,19]}
import json
import openai

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="<your-api-key>",  # Available from https://wandb.ai/authorize
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
        {"role": "user", "content": "Give me a list of three fruits with their colors."},
    ],
    response_format={"type": "json_object"}  # This enables JSON mode
)

content = response.choices[0].message.content
parsed = json.loads(content)
print(parsed)
```

{{% /tab %}}
{{% tab header="Bash" value="bash" %}}

```bash {hl_lines=[10]}
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
        {"role": "user", "content": "Give me a list of three fruits with their colors."},
    ],
    "response_format": {"type": "json_object"}
  }'
```

{{% /tab %}}
{{< /tabpane >}}
