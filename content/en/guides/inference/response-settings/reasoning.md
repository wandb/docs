---
description: >-
  How to return and view reasoning in your W&B Inference responses.
displayed_sidebar: default
title: "View reasoning information"
weight: 2
---

Reasoning models, like [OpenAI's GPT OSS 20B](https://huggingface.co/openai/gpt-oss-20b), include information about their reasoning steps as part of the output returned in addition to the final answer. This is automatic and no additional input parameters are needed.

You can determine whether a model supports reasoning or not by checking the Supported Features sections of its catalog page in the UI.

You can find reasoning information in the a `reasoning_content` field of responses. This field is not present in the outputs of other models.

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
