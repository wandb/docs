---
title: "W&B Inference"
weight: 8
description: >
  Access open-source foundation models through W&B Weave and an OpenAI-compatible API
---

W&B Inference gives you access to leading open-source foundation models through W&B Weave and an OpenAI-compatible API. You can:

- Build AI applications and agents without signing up for a hosting provider or self-hosting a model
- Try [supported models]({{< relref "models" >}}) in the W&B Weave Playground

With Weave, you can trace, evaluate, monitor, and improve your W&B Inference-powered applications.

## Quickstart

Here's a simple example using Python:

```python
import openai

client = openai.OpenAI(
    # The custom base URL points to W&B Inference
    base_url='https://api.inference.wandb.ai/v1',
    
    # Get your API key from https://wandb.ai/authorize
    api_key="<your-api-key>",
    
    # Team and project are required for usage tracking
    project="<your-team>/<your-project>",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
)

print(response.choices[0].message.content)
```

## Next steps

1. Review the [available models]({{< relref "models" >}}) and [usage information and limits]({{< relref "usage-limits" >}})
2. Set up your account using the [prerequisites]({{< relref "prerequisites" >}})
3. Use the service through the [API]({{< relref "api-reference" >}}) or [UI]({{< relref "ui-guide" >}})
4. Try the [usage examples]({{< relref "examples" >}})

## Usage details

{{< alert title="Important" color="warning" >}}
W&B Inference credits come with Free, Pro, and Academic plans for a limited time. Availability may vary for Enterprise accounts. When credits run out:

- **Free users** must upgrade to a paid plan to continue using Inference.  
  👉 [Upgrade to Pro or Enterprise](https://wandb.ai/subscriptions)
- **Pro users** are billed monthly for usage beyond free credits, up to a default cap of $6,000/month. See [Account tiers and default usage caps]({{< relref "usage-limits#account-tiers-and-default-usage-caps" >}})
- **Enterprise usage** is capped at $700,000/year. Your account executive handles billing and limit increases. See [Account tiers and default usage caps]({{< relref "usage-limits#account-tiers-and-default-usage-caps" >}})

To learn more, visit the [pricing page](https://wandb.ai/site/pricing/) or see [model-specific costs](https://wandb.ai/site/pricing/inference).
{{< /alert >}}