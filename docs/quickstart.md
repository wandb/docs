---
description: W&B Quickstart.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quickstart


Install W&B and start tracking your machine learning experiments in minutes.




## Create an account and install W&B
Complete these steps to use W&B:

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

<!-- Next, log in to W&B:

```
wandb login
```

Or if you're using [W&B Server:](./guides/hosting/intro.md)

```
wandb login --host=http://wandb.your-shared-local-host.com
``` -->

  </TabItem>
  <TabItem value="notebook">

Install the CLI and Python library for interacting with the Weights and Biases API:

```python
!pip install wandb
```

<!-- Next, import the W&B Python SDK and log in:

```python
import wandb
wandb.login()
``` -->

  </TabItem>
</Tabs>

## Log in to W&B


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

Or if you're using [W&B Server:](./guides/hosting/intro.md)

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


## Start a W&B Run

Initialize a W&B Run object in your Python script or notebook with [`wandb.init()`](./ref/python/run.md).

```python
run = wandb.init(project="my-awesome-project")
```

 A run is the basic building block of W&B. You will use them often to track metrics, create logs, create jobs, and more.

## Track metrics and hyperparameters

Use [`wandb.log()`](./ref/python/log.md) to track metrics:

```python
wandb.log({'accuracy': acc, 'loss': loss})
```

Anything you log with `wandb.log` is stored in the run object that was most recently initialized.


## Putting it all together

```python
# train.py

import wandb
import random # for demo script

epochs=10
lr=0.01

run = wandb.init(
    # Set the project where this run will be logged
    project="job_example",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    })

offset = random.random() / 5
print(f"lr: {lr}")

for epoch in range(2, epochs):
    # simulating a training run
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})

run.log_code()
```

<!-- ## Track hyperparameters

Save hyperparameters with [`wandb.config`](./guides/track/config.md). Tracking hyperparameters makes it easy to compare experiments with the W&B App. 

```python
wandb.config.dropout = 0.2
```
Attributes stored in a `wandb.config` object are associated with the most recent initialized Run object.

![](/images/quickstart/wandb_demo_experiments.gif) -->

<!-- ## Get alerts

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


## What next?

![](/images/quickstart/wandb_demo_logging_metrics.png)

<!-- Set up Alerts! -->
<!-- view what was tracked -->
<!-- or a framework [integration](guides/integrations/intro.md) for easy instrumentation. -->


1. [**Collaborative Reports**](./guides/reports/intro.md): Snapshot results, take notes, and share findings
2. [**Data + Model Versioning**](./guides/data-and-model-versioning/intro.md): Track dependencies and results in your ML pipeline
3. [**Data Visualization**](guides/data-vis/intro.md): Visualize and query datasets and model evaluations
4. [**Hyperparameter Tuning**](guides/sweeps/intro.md): Quickly automate optimizing hyperparameters
5. [**Private-Hosting**](guides/hosting/intro.md): The enterprise solution for private cloud or on-prem hosting of W&B

## Common Questions

**Where do I find my API key?**
Once you've signed in to www.wandb.ai, the API key will be on the [Authorize page](https://wandb.ai/authorize).

**How do I use W&B in an automated environment?**
If you are training models in an automated environment where it's inconvenient to run shell commands, such as Google's CloudML, you should look at our guide to configuration with [Environment Variables](guides/track/environment-variables.md).

**Do you offer local, on-prem installs?**
Yes, you can [privately host W&B](guides/hosting/intro.md) locally on your own machines or in a private cloud, try [this quick tutorial notebook](http://wandb.me/intro) to see how. Note, to login to wandb local server you can [set the host flag](./guides/hosting/basic-setup.md#login) to the address of the local instance.  **** 

**How do I turn off wandb logging temporarily?**
If you're testing code and want to disable wandb syncing, set the environment variable [`WANDB_MODE=offline`](guides/track/environment-variables.md).
