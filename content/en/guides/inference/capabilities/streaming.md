---
description: >-
  How to use streaming output with W&B Inference.
displayed_sidebar: default
title: "Streaming responses"
weight: 1
---

Sometimes models take a while to generate a response.
Setting the `stream` option to true allows you to receive the response as a stream
of chunks, allowing you to incrementally display results instead of waiting for the entire
response to be generated.

Streaming output is supported for all hosted models. We especially encourage its
use with [reasoning models](./reasoning), as non-streaming requests may timeout if the model thinks for
too long before output starts.

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python {hl_lines=[13,"16-20"]}
import openai

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="<your-api-key>",  # Available from https://wandb.ai/authorize
)

stream = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Tell me a rambling joke"}
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    else:
        print(chunk) # Show CompletionUsage object
```

{{% /tab %}}
{{% tab header="Bash" value="bash" %}}

```bash {hl_lines=[9]}
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      { "role": "user", "content": "Tell me a rambling joke" }
    ],
    "stream": true
  }'
```

{{% /tab %}}
{{< /tabpane >}}
