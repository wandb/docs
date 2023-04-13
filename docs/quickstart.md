---
description: W&B Quickstart.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quickstart

Install W&B and start tracking your machine learning experiments in minutes.

## 1. Create an account and install W&B
Before you get started, make sure you create an account and install W&B:

1. [Sign up](https://wandb.ai/site) for a free account at [https://wandb.ai/site](https://wandb.ai/site) and then login to your wandb account.  
2. Install the wandb library on your machine in a Python 3 environment using [`pip`](https://pypi.org/project/wandb/).  
<!-- 3. Login to the wandb library on your machine. You will find your API key here: [https://wandb.ai/authorize](https://wandb.ai/authorize).   -->

The following code snippets demonstrate how to install and log into W&B using the W&B CLI and Python Library:

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

Install the CLI and Python library for interacting with the Weights and Biases API:

```
pip install wandb
```

  </TabItem>
  <TabItem value="notebook">

Install the CLI and Python library for interacting with the Weights and Biases API:

```python
!pip install wandb
```


  </TabItem>
</Tabs>

## 2. Log in to W&B


<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

Next, log in to W&B:

```
wandb login
```

Or if you're using [W&B Server:](./guides/hosting)

```
wandb login --host=http://wandb.your-shared-local-host.com
```

Provide [your API key](https://wandb.ai/authorize) when prompted.

  </TabItem>
  <TabItem value="notebook">

Next, import the W&B Python SDK and log in:

```python
wandb.login()
```

Provide [your API key](https://wandb.ai/authorize) when prompted.
  </TabItem>
</Tabs>


## 3. Start a  run and track hyperparameters

Initialize a W&B Run object in your Python script or notebook with [`wandb.init()`](./ref/python/run.md) and pass a dictionary to the `config` parameter with key-value pairs of hyperparameter names and values:

```python
run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    })
```


<!-- ```python
run = wandb.init(project="my-awesome-project")
``` -->

A [run](./guides/runs) is the basic building block of W&B. You will use them often to [track metrics](./guides/track), [create logs](./guides/artifacts), [create jobs](./guides/launch), and more.


<!-- ## Track metrics -->
<!-- Pass a dictionary to the `config` parameter with key-value pairs of hyperparameter name and values when you initialize a run object:

```python
  # Track hyperparameters and run metadata
  config={
      "learning_rate": lr,
      "epochs": epochs,
  }
``` -->


<!-- Use [`wandb.log()`](./ref/python/log.md) to track metrics:

```python
wandb.log({'accuracy': acc, 'loss': loss})
```

Anything you log with `wandb.log` is stored in the run object that was most recently initialized. -->



## Putting it all together

Putting it all together, your training script might look similar to the following code example. The highlighted code shows W&B-specific code. 
Note that we added code that mimics machine learning training.

```python
# train.py
import wandb
import random # for demo script

# highlight-next-line
wandb.login()

epochs=10
lr=0.01

# highlight-start
run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    })
# highlight-end    

offset = random.random() / 5
print(f"lr: {lr}")

# simulating a training run
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # highlight-next-line
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

That's it! Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home) to view how the metrics we logged with W&B (accuracy and loss) improved during each training step.

![Shows the loss and accuracy that was tracked from each time we ran the script above. ](/images/quickstart/quickstart_image.png)

The image above (click to expand) shows the loss and accuracy that was tracked from each time we ran the script above.  Each run object that was created is show within the **Runs** column. Each run name is randomly generated.


## What's next?
<!-- ### Get alerts

Get notified by Slack or email if your W&B Run has crashed or with a custom trigger. For example, you can create a trigger to notify you if your loss reports `NaN` or a step in your ML pipeline has completed.

Follow the procedure outlined below to set up an alert: 

1. Turn on Alerts in your W&B [User Settings](https://wandb.ai/settings).
2. Add [`wandb.alert()`](./guides/runs/alert.md) to your code.

```python
wandb.alert(
    title="Low accuracy", 
    text=f"Accuracy {acc} is below threshold {thresh}"
)
```
You will receive an email or Slack alert when your alert criteria is met. For example, the proceeding image demonstrates a Slack alert:

![W&B Alerts in a Slack channel](/images/quickstart/get_alerts.png)

See the [Alerts docs](./guides/runs/alert.md) for more information on how to set up an alert. For more information about setting options, see the [Settings](./guides/app/settings-page/intro.md) page.  -->

Explore the rest of the W&B ecosystem.

1. Check out [W&B Integrations](guides/integrations) to learn how to integrate W&B with your ML framework such as PyTorch, ML library such as Hugging Face, or ML service such as SageMaker. 
2. Organize runs, embed and automate visualizations, describe your findings, and share updates with collaborators with [W&B Reports](./guides/reports).
2. Create [W&B Artifacts](./guides/artifacts) to track datasets, models, dependencies, and results through each step of your machine learning pipeline.
3. Automate hyperparameter search and explore the space of possible models with [W&B Sweeps](./guides/sweeps).
4. Understand your datasets, visualize model predictions, and share insights in a [central dashboard](./guides/data-vis).


![](/images/quickstart/wandb_demo_experiments.gif) 



## Common Questions

**Where do I find my API key?**
Once you've signed in to www.wandb.ai, the API key will be on the [Authorize page](https://wandb.ai/authorize).

**How do I use W&B in an automated environment?**
If you are training models in an automated environment where it's inconvenient to run shell commands, such as Google's CloudML, you should look at our guide to configuration with [Environment Variables](guides/track/environment-variables).

**Do you offer local, on-prem installs?**
Yes, you can [privately host W&B](guides/hosting) locally on your own machines or in a private cloud, try [this quick tutorial notebook](http://wandb.me/intro) to see how. Note, to login to wandb local server you can [set the host flag](guides/hosting/how-to-guides/basic-setup) to the address of the local instance.  

**How do I turn off wandb logging temporarily?**
If you're testing code and want to disable wandb syncing, set the environment variable [`WANDB_MODE=offline`](guides/track/environment-variables).
