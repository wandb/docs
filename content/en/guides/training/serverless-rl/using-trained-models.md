---
title: "Use your trained models"
linkTitle: "Use trained models"
weight: 20
description: >
  Make inference requests to the models you've trained.
---

After training a model with Serverless RL, it is automatically deployed and ready to use in production. To make a request to the OpenAI-compatible endpoint, follow the instructions below.

### Overview

When making inference requests, set your base URL to `https://api.training.wandb.ai/v1/` and use your W&B API key for authentication. Finally, set your model name to match the following schema:

`wandb-artifact:///<entity>/<project>/<model-name>:<step>`

For example, if your W&B team is named `email-specialists`, your project is called `mail-search`, your trained model is named `agent-001`, and you wanted to deploy it on step 25 (presumably because that step did the best on your evals), you would set the model name in your chat completion requests to the following:

`wandb-artifact:///email-specialists/mail-search/agent-001:step25`

This tells W&B Inference exactly where to find your model when initially loading it for inference.

The following examples show you how to make inference requests to your trained model using a CuRL request or the [Python OpenAI SDK](https://openai.github.io/openai-agents-python/).

### CuRL

```shell
curl https://api.training.wandb.ai/v1/chat/completions \
    -H "Authorization: Bearer $WANDB_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
            "model": "wandb-artifact:///<entity>/<project>/<model-name>:<step>",
            "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize our training run."}
            ],
            "temperature": 0.7,
            "top_p": 0.95
        }'
```

### OpenAI SDK

```python
from openai import OpenAI

WANDB_API_KEY = "your-wandb-api-key"
ENTITY = "my-entity"
PROJECT = "my-project"

client = OpenAI(
    base_url="https://api.inference.coreweave.com/v1",
    api_key=WANDB_API_KEY,
    default_headers={"OpenAI-Project": f"{ENTITY}/{PROJECT}"},
)

response = client.chat.completions.create(
    model=f"wandb-artifact:///{ENTITY}/{PROJECT}/my-model:step100",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize our training run."},
    ],
    temperature=0.7,
    top_p=0.95,
)

print(response.choices[0].message.content)
```