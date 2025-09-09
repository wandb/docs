---
description: W&B Quickstart
menu:
  default:
    identifier: ja-guides-quickstart
    parent: guides
title: W&B Quickstart
url: quickstart
weight: 2
---

Install W&B to track, visualize, and manage machine learning experiments of any size.

{{% alert %}}
Are you looking for information on W&B Weave? See the [Weave Python SDK quickstart](https://weave-docs.wandb.ai/quickstart) or [Weave TypeScript SDK quickstart](https://weave-docs.wandb.ai/reference/generated_typescript_docs/intro-notebook).
{{% /alert %}}

## Sign up and create an API key

To authenticate your machine with W&B, generate an API key from your user profile or at [wandb.ai/authorize](https://wandb.ai/authorize). Copy the API key and store it securely.

## Install the `wandb` library and log in

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. Set the `WANDB_API_KEY` [environment variable]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}).

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. Install the `wandb` library and log in.

    ```shell
    pip install wandb
    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb

wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb
import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## Start a run and track hyperparameters

In your Python script or notebook, initialize a W&B run object with [`wandb.init()`]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}). Use a dictionary for the `config` parameter to specify hyperparameter names and values.

```python
run = wandb.init(
    project="my-awesome-project",  # Specify your project
    config={                        # Track hyperparameters and metadata
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

A [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) serves as the core element of W&B, used to [track metrics]({{< relref path="/guides/models/track/" lang="ja" >}}), [create logs]({{< relref path="/guides/models/track/log/" lang="ja" >}}), and more.

## Assemble the components

This mock training script logs simulated accuracy and loss metrics to W&B:

```python
import wandb
import random

wandb.login()

# Project that the run is recorded to
project = "my-awesome-project"

# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    offset = random.random() / 5
    print(f"lr: {config['lr']}")
    
    # Simulate a training run
    for epoch in range(2, config['epochs']):
        acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
        loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
        print(f"epoch={config['epochs']}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})
```

Visit [wandb.ai/home](https://wandb.ai/home) to view recorded metrics such as accuracy and loss and how they changed during each training step. The following image shows the loss and accuracy tracked from each run. Each run object appears in the **Runs** column with generated names.

{{< img src="/images/quickstart/quickstart_image.png" alt="Shows loss and accuracy tracked from each run." >}}

## Next steps

Explore more features of the W&B ecosystem:

1. Read the [W&B Integration tutorials]({{< relref path="guides/integrations/" lang="ja" >}}) that combine W&B with frameworks like PyTorch, libraries like Hugging Face, and services like SageMaker.
2. Organize runs, automate visualizations, summarize findings, and share updates with collaborators using [W&B Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}).
3. Create [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) to track datasets, models, dependencies, and results throughout your machine learning pipeline.
4. Automate hyperparameter searches and optimize models with [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}).
5. Analyze runs, visualize model predictions, and share insights on a [central dashboard]({{< relref path="/guides/models/tables/" lang="ja" >}}).
6. Visit [W&B AI Academy](https://wandb.ai/site/courses/) to learn about LLMs, MLOps, and W&B Models through hands-on courses.
7. Visit [weave-docs.wandb.ai](https://weave-docs.wandb.ai/) to learn how to track track, experiment with, evaluate, deploy, and improve your LLM-based applications using Weave.