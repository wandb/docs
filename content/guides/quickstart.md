---
description: W&B Quickstart
menu:
  default:
    identifier: quickstart_models
    parent: guides
title: W&B Quickstart
url: quickstart
weight: 2
---
Install W&B and track machine learning experiments in minutes.

## Sign up and create an API key

An API key authenticates your machine to W&B. Generate an API key from your user profile.

{{% alert %}}
To streamline the process, generate an API key directly at [https://wandb.ai/authorize](https://wandb.ai/authorize). Copy the displayed API key and save it securely, such as in a password manager.
{{% /alert %}}

1. Click your user profile icon in the upper right corner.
1. Select **User Settings**, then scroll to the **API Keys** section.
1. Click **Reveal**, and copy the displayed API key. To hide the API key, reload the page.

## Install the `wandb` library and log in

To install the `wandb` library locally and log in:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. Set the `WANDB_API_KEY` [environment variable]({{< relref "/guides/models/track/environment-variables.md" >}}) to your API key.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. Install the `wandb` library and log in.

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

Initialize a W&B Run object in your Python script or notebook with [`wandb.init()`]({{< relref "/ref/python/run.md" >}}) and pass a dictionary to the `config` parameter containing hyperparameter name-value pairs:

```python
run = wandb.init(
    # Set the project where this run logs
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

A [run]({{< relref "/guides/models/track/runs/" >}}) serves as the basic building block of W&B. Use runs to [track metrics]({{< relref "/guides/models/track/" >}}), [create logs]({{< relref "/guides/core/artifacts/" >}}), and more.

## Put it all together

Your training script may look similar to the following example.

```python
# train.py
import wandb
import random  # for demo script

wandb.login()

epochs = 10
lr = 0.01

run = wandb.init(
    # Set the project where this run logs
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)

offset = random.random() / 5
print(f"lr: {lr}")

# Simulating a training run
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home) to view metrics logged with W&B (accuracy and loss) during each training step.

{{< img src="/images/quickstart/quickstart_image.png" alt="Shows the loss and accuracy tracked from each run of the script above." >}}

The image above (click to expand) displays the loss and accuracy tracked during each run of the script. Each created run object appears in the **Runs** column, with each run name randomly generated.

## What's next?

Explore the W&B ecosystem further.

1. Check [W&B Integrations]({{< relref "guides/integrations/" >}}) to learn how to integrate W&B with ML frameworks like PyTorch, ML libraries like Hugging Face, or ML services like SageMaker.
2. Organize runs, embed and automate visualizations, describe findings, and share updates with collaborators using [W&B Reports]({{< relref "/guides/core/reports/" >}}).
3. Create [W&B Artifacts]({{< relref "/guides/core/artifacts/" >}}) to track datasets, models, dependencies, and results at each step of the machine learning pipeline.
4. Automate hyperparameter searches and explore model possibilities with [W&B Sweeps]({{< relref "/guides/models/sweeps/" >}}).
5. Analyze datasets, visualize model predictions, and share insights on a [central dashboard]({{< relref "/guides/models/tables/" >}}).
6. Visit W&B AI Academy for hands-on [courses](https://wandb.me/courses) on LLMs, MLOps, and W&B Models.