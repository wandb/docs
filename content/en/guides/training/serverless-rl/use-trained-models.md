---
title: "Use your trained models"
linkTitle: "Use trained models"
weight: 20
description: Make inference requests to the models you've trained.
---

After training a model with Serverless RL, it is automatically available for inference.

To send requests to your trained model, you need:
* Your [W&B API key](https://wandb.ai/authorize)
* The [Training API's]({{< relref "/ref/training" >}}) base URL, `https://api.training.wandb.ai/v1/`
* Your model's endpoint

The model's endpoint uses the following schema:

```
wandb-artifact:///<entity>/<project>/<model-name>:<step>
```

The schema consists of:

* Your W&B entity's (team) name
* The name of the project associated with your model
* The trained model's name
* The training step of the model you want to deploy (this is usually the step where the model performed best in your evaluations)

For example, if your W&B team is named `email-specialists`, your project is called `mail-search`, your trained model is named `agent-001`, and you wanted to deploy it on step 25, the endpoint looks like this:

```
wandb-artifact:///email-specialists/mail-search/agent-001:step25
```

Once you have your endpoint, you can integrate it into your normal inference workflows. The following examples show how to make inference requests to your trained model using a cURL request or the [Python OpenAI SDK](https://github.com/openai/openai-python).

### cURL

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
    base_url="https://api.training.wandb.ai/v1",
    api_key=WANDB_API_KEY
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