import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Quickstart

Build better models more efficiently with Weights & Biases experiment tracking.

### [Run a quick example project →](http://wandb.me/intro)

Try this short Google Colab to see Weights & Biases in action, no code installation required!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/intro)

![](<./images/quickstart/wandb_demo_experiments.gif>)

### 1. Set up wandb

**a)** [Sign up](https://wandb.ai/site) for a free account at [https://wandb.ai/site](https://wandb.ai/site) and then login to your wandb account.

**b)** Install the wandb library on your machine in a Python 3 environment using `pip`

**c)** Login to the wandb library on your machine. You will find your API key here: [https://wandb.ai/authorize](https://wandb.ai/authorize).

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

Install the CLI and Python library for interacting with the Weights and Biases API:

```
pip install wandb
```

Next, log in to W&B:

```
wandb login
```

Or if you're using [W&B Server:](./guides/hosting/intro.md)

```
wandb login --host=http://wandb.your-shared-local-host.com
```

  </TabItem>
  <TabItem value="notebook">

Install the CLI and Python library for interacting with the Weights and Biases API:

```python
!pip install wandb
```

Next, import the W&B Python SDK and log in:

```python
import wandb
wandb.login()
```

  </TabItem>
</Tabs>


### 2. Start a new run

Initialize a new run in W&B in your Python script or notebook. `wandb.init()` will start tracking system metrics and console logs, right out of the box. Run your code, put in [your API key](https://wandb.ai/authorize) when prompted, and you'll see the new run appear in W&B.[More about wandb.init() →](guides/track/launch.md)

```python
import wandb
wandb.init(project="my-awesome-project")
```

### 3. Track metrics

Use `wandb.log()` to track metrics or a framework [integration](guides/integrations/intro.md) for easy instrumentation.
[More about wandb.log() →](guides/track/log/intro.md)

```python
wandb.log({'accuracy': train_acc, 'loss': train_loss})
```

![](<./images/quickstart/wandb_demo_logging_metrics.png>)

### 4. Track hyperparameters

Save hyperparameters so you can quickly compare experiments.\
[More about wandb.config →](guides/track/config.md)

```python
wandb.config.dropout = 0.2
```
![](<./images/quickstart/wandb_demo_logging_config.png>)

### 5. Get alerts

Get notified via Slack or email if your W&B Run has crashed or whether a custom trigger, such as your loss going to NaN or a step in your ML pipeline has completed, has been reached. See the [Alerts docs](https://docs.wandb.ai/guides/track/alert) for a full setup.

[More about wandb.alert() →](./guides/track/advanced/alert.md)

1. Turn on Alerts in your W&B [User Settings](./guides/app/settings-page/intro.md)
2. Add `wandb.alert()` to your code

```python
wandb.alert(
    title="Low accuracy", 
    text=f"Accuracy {acc} is below the acceptable threshold {thresh}"
)
```

Then see W&B Alerts messages in Slack (or your email):

![W&B Alerts in a Slack channel](<./images/quickstart/get_alerts.png>)

## What next?

1. [**Collaborative Reports**](./guides/reports/intro.md): Snapshot results, take notes, and share findings
2. [**Data + Model Versioning**](./guides/models/intro.md): Track dependencies and results in your ML pipeline
3. [**Data Visualization**](guides/data-vis/intro.md): Visualize and query datasets and model evaluations
4. [**Hyperparameter Tuning**](guides/sweeps/intro.md): Quickly automate optimizing hyperparameters
5. ****[**Private-Hosting**](guides/hosting/intro.md): The enterprise solution for private cloud or on-prem hosting of W&B

## Common Questions

**Where do I find my API key?**
Once you've signed in to www.wandb.ai, the API key will be on the [Authorize page](https://wandb.ai/authorize).

**How do I use W&B in an automated environment?**
If you are training models in an automated environment where it's inconvenient to run shell commands, such as Google's CloudML, you should look at our guide to configuration with [Environment Variables](guides/track/advanced/environment-variables.md).

**Do you offer local, on-prem installs?**
Yes, you can [privately host W&B](guides/hosting/intro.md) locally on your own machines or in a private cloud, try [this quick tutorial notebook](http://wandb.me/intro) to see how. Note, to login to wandb local server you can [set the host flag](https://docs.wandb.ai/guides/hosting/quickstart#4.-modify-training-code-to-log-to-wandb-local-server) to the address of the local instance.  **** 

**How do I turn off wandb logging temporarily?**\
If you're testing code and want to disable wandb syncing, set the environment variable [`WANDB_MODE=offline`](guides/track/advanced/environment-variables.md).
