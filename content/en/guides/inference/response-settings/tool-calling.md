---
description: >-
  How to use Tool Calling with W&B Inference.
displayed_sidebar: default
title: "Use tool calling"
weight: 5
---

Tool calling allows you to extend a model's capabilities to include invoking tools as part of its response. W&B Inference only supports calling functions at this time. 

To call functions, specify them and their arguments as part of your request to the model. The model determines whether it needs to run the function to fulfill a request, and then specifies the function's argument values if needed.


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
