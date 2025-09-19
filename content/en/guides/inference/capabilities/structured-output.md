---
description: >-
  How to use Structured Output with W&B Inference.
displayed_sidebar: default
title: "Structured output"
weight: 4
---

Structured Output is similar to [JSON mode](../json-mode) but provides the added benefit
of ensuring that the model's response adheres to the schema you specify.
We recommend using Structured Output instead of JSON mode when possible.

To enable Structured Output, specify "json_schema" as the "response_format" type in the request.

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python {hl_lines=["15-31"]}
import json
import openai

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="<your-api-key>",  # Available from https://wandb.ai/authorize
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "CalendarEventResponse",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "participants": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "date", "participants"],
                "additionalProperties": False,
            },
        },
    },
)

content = response.choices[0].message.content
parsed = json.loads(content)
print(parsed)
```

{{% /tab %}}
{{% tab header="Bash" value="bash" %}}

```bash {hl_lines=["10-26"]}
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "CalendarEventResponse",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "participants": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "date", "participants"],
                "additionalProperties": False,
            },
        },
    },
  }'
```

{{% /tab %}}
{{< /tabpane >}}
