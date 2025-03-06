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
Install W&B and start tracking your machine learning experiments in minutes.

## Sign up and create an API key

An API key authenticates your machine to W&B. You can generate an API key from your user profile.

{{% alert %}}
For a more streamlined approach, you can generate an API key by going directly to [https://wandb.ai/authorize](https://wandb.ai/authorize). Copy the displayed API key and save it in a secure location such as a password manager.
{{% /alert %}}

1. Click your user profile icon in the upper right corner.
1. Select **User Settings**, then scroll to the **API Keys** section.
1. Click **Reveal**. Copy the displayed API key. To hide the API key, reload the page.

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

Initialize a W&B Run object in your Python script or notebook with [`wandb.init()`]({{< relref "/ref/python/run.md" >}}) and pass a dictionary to the `config` parameter with key-value pairs of hyperparameter names and values:

```python
run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```


A [run]({{< relref "/guides/models/track/runs/" >}}) is the basic building block of W&B. You will use them often to [track metrics]({{< relref "/guides/models/track/" >}}), [create logs]({{< relref "/guides/core/artifacts/" >}}), and more.


## Put it all together

Putting it all together, your training script might look similar to the following code example. The highlighted code shows W&B-specific code. 
Note that we added code that mimics machine learning training.

```python
# train.py
import wandb
import random  # for demo script

# highlight-next-line
wandb.login()

epochs = 10
lr = 0.01

# highlight-start
run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)
# highlight-end

offset = random.random() / 5
print(f"lr: {lr}")

# simulating a training run
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # highlight-next-line
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

That's it. Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home) to view how the metrics we logged with W&B (accuracy and loss) improved during each training step.

{{< img src="/images/quickstart/quickstart_image.png" alt="Shows the loss and accuracy that was tracked from each time we ran the script above. " >}}

The image above (click to expand) shows the loss and accuracy that was tracked from each time we ran the script above. Each run object that was created is show within the **Runs** column. Each run name is randomly generated.


## What's next?

Explore the rest of the W&B ecosystem.

1. Check out [W&B Integrations]({{< relref "guides/integrations/" >}}) to learn how to integrate W&B with your ML framework such as PyTorch, ML library such as Hugging Face, or ML service such as SageMaker. 
2. Organize runs, embed and automate visualizations, describe your findings, and share updates with collaborators with [W&B Reports]({{< relref "/guides/core/reports/" >}}).
2. Create [W&B Artifacts]({{< relref "/guides/core/artifacts/" >}}) to track datasets, models, dependencies, and results through each step of your machine learning pipeline.
3. Automate hyperparameter search and explore the space of possible models with [W&B Sweeps]({{< relref "/guides/models/sweeps/" >}}).
4. Understand your datasets, visualize model predictions, and share insights in a [central dashboard]({{< relref "/guides/models/tables/" >}}).
5. Navigate to W&B AI Academy and learn about LLMs, MLOps and W&B Models from hands-on [courses](https://wandb.me/courses).
