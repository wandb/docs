---
description: >-
  How to use Tool Calling with W&B Inference.
displayed_sidebar: default
title: "Tool calling"
weight: 5
---

Tool Calling allows you to extend a model's capabilities to include invoking tools as part of its response. The only kind of Tool Calling currently supported within W&B Inference is Function Calling.

With Function Calling, you specify available functions and their arguments as part of your request to the model. The model will determine if the request should be fulfilled by running that function, and what the argument values should be.


{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python {hl_lines=["13-30", 32]}
import openai

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="<your-api-key>",  # Available from https://wandb.ai/authorize
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "What is the weather like in San Francisco? Use Fahrenheit."},
    ],
    tool_choice="auto",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            },
        }
    ],
)

print(response.choices[0].message.tool_calls)
```

{{% /tab %}}
{{% tab header="Bash" value="bash" %}}

```bash {hl_lines=["9-26"]}
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "What is the weather like in San Francisco? Use Fahrenheit."},
        ],
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location", "unit"],
                    },
                },
            }
        ],
  }'
```

{{% /tab %}}
{{< /tabpane >}}
