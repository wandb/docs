---
description: >-
  How to use Reasoning with W&B Inference.
displayed_sidebar: default
title: "Reasoning"
weight: 2
---

Reasoning models like OpenAI GPT OSS 20B will include information about their reasoning steps as part of the output
returned in addition to the final answer. This is automatic; no additional input parameters are needed.

The "Supported Features" section of each model's catalog page in the UI will indicate whether reasoning is supported for that model.

The additional information will be in a `reasoning_content` field. This field is not present in the outputs of other models.

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python {hl_lines=[15]}
import openai

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="<your-api-key>",  # Available from https://wandb.ai/authorize
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "3.11 and 3.8, which is greater?"}
    ],
)

print(response.choices[0].message.reasoning_content)
print("--------------------------------")
print(response.choices[0].message.content)
```

{{% /tab %}}
{{% tab header="Bash" value="bash" %}}

```bash
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      { "role": "user", "content": "3.11 and 3.8, which is greater?" }
    ],
  }'
```

{{% /tab %}}
{{< /tabpane >}}
