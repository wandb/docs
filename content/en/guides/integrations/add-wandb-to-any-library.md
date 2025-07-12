---
menu:
  default:
    identifier: add-wandb-to-any-library
    parent: integrations
title: Add wandb to any library
weight: 10
---

## Add wandb to any library

This guide provides best practices on how to integrate W&B into your Python library to get powerful Experiment Tracking, GPU and System Monitoring, Model Checkpointing, and more for your own library.

{{% alert %}}
If you are still learning how to use W&B, we recommend exploring the other W&B Guides in these docs, such as [Experiment Tracking]({{< relref "/guides/models/track" >}}), before reading further.
{{% /alert %}}

Below we cover best tips and best practices when the codebase you are working on is more complicated than a single Python training script or Jupyter notebook. The topics covered are:

* Setup requirements
* User Login
* Starting a wandb Run
* Defining a Run Config
* Logging to W&B
* Distributed Training
* Model Checkpointing and More
* Hyper-parameter tuning
* Advanced Integrations

### Setup requirements

Before you get started, decide whether or not to require W&B in your library’s dependencies:

#### Require W&B on installation

Add the W&B Python library (`wandb`) to your dependencies file, for example, in your `requirements.txt` file:

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### Make W&B optional on installation

There are two ways to make the W&B SDK (`wandb`) optional:

A. Raise an error when a user tries to use `wandb` functionality without installing it manually and show an appropriate error message:

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "You are trying to use wandb which is not currently installed."
        "Please install it using pip install wandb"
    ) 
```

B. Add `wandb` as an optional dependency to your `pyproject.toml` file, if you are building a Python package:

```toml
[project]
name = "my_awesome_lib"
version = "0.1.0"
dependencies = [
    "torch",
    "sklearn"
]

[project.optional-dependencies]
dev = [
    "wandb"
]
```

### User login

#### Create an API key

An API key authenticates a client or machine to W&B. You can generate an API key from your user profile.

{{% alert %}}
For a more streamlined approach, you can generate an API key by going directly to [https://wandb.ai/authorize](https://wandb.ai/authorize). Copy the displayed API key and save it in a secure location such as a password manager.
{{% /alert %}}

1. Click your user profile icon in the upper right corner.
1. Select **User Settings**, then scroll to the **API Keys** section.
1. Click **Reveal**. Copy the displayed API key. To hide the API key, reload the page.

#### Install the `wandb` library and log in

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

{{% tab header="Python notebook" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

If a user is using wandb for the first time without following any of the steps mentioned above, they will automatically be prompted to log in when your script calls `wandb.init`.

### Start a run

A W&B Run is a unit of computation logged by W&B. Typically, you associate a single W&B Run per training experiment.

Initialize W&B and start a Run within your code with:

```python
run = wandb.init()
```

Optionally, you can provide a name for their project, or let the user set it themselves with parameters such as `wandb_project` in your code along with the username or team name, such as `wandb_entity`, for the entity parameter:

```python
run = wandb.init(project=wandb_project, entity=wandb_entity)
```

You must call `run.finish()` to finish the run. If this works with your integration's design,  use the run as a context manager:

```python
# When this block exits, it calls run.finish() automatically.
# If it exits due to an exception, it uses run.finish(exit_code=1) which
# marks the run as failed.
with wandb.init() as run:
    ...
```


#### When to call `wandb.init`?

Your library should create W&B Run as early as possible because any output in your console, including error messages, is logged as part of the W&B Run. This makes debugging easier.

#### Use `wandb` as an optional dependency

If you want to make `wandb` optional when your users use your library, you can either:

* Define a `wandb` flag such as:

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python
trainer = my_trainer(..., use_wandb=True)
```
{{% /tab %}}

{{% tab header="Bash" value="bash" %}}

```bash
python train.py ... --use-wandb
```
{{% /tab %}}

{{< /tabpane >}}

* Or, set `wandb` to be `disabled` in `wandb.init`:

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python
wandb.init(mode="disabled")
```
{{% /tab %}}

{{% tab header="Bash" value="bash" %}}

```bash
export WANDB_MODE=disabled
```

or

```bash
wandb disabled
```
{{% /tab %}}

{{< /tabpane >}}

* Or, set `wandb` to be offline - note this will still run `wandb`, it just won't try and communicate back to W&B over the internet:

{{< tabpane text=true >}}

{{% tab header="Environment Variable" value="environment" %}}

```bash
export WANDB_MODE=offline
```

or

```python
os.environ['WANDB_MODE'] = 'offline'
```
{{% /tab %}}

{{% tab header="Bash" value="bash" %}}

```bash
wandb offline
```
{{% /tab %}}

{{< /tabpane >}}

### Define a run config
With a `wandb` run config, you can provide metadata about your model, dataset, and so on when you create a W&B Run. You can use this information to compare different experiments and quickly understand the main differences.

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs table" >}}

Typical config parameters you can log include:

* Model name, version, architecture parameters, etc.
* Dataset name, version, number of train/val examples, etc.
* Training parameters such as learning rate, batch size, optimizer, etc.

The following code snippet shows how to log a config:

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Update the run config
Use `run.config.update` to update the config. Updating your configuration dictionary is useful when parameters are obtained after the dictionary was defined. For example, you might want to add a model’s parameters after the model is instantiated.

```python
run.config.update({"model_parameters": 3500})
```

For more information on how to define a config file, see [Configure experiments]({{< relref "/guides/models/track/config" >}}).

### Log to W&B

#### Log metrics

Create a dictionary where the key value is the name of the metric. Pass this dictionary object to [`run.log`]({{< relref "/guides/models/track/log" >}}):

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        run.log(metrics)
```

If you have a lot of metrics, you can have them automatically grouped in the UI by using prefixes in the metric name, such as `train/...` and `val/...`. This will create separate sections in your W&B Workspace for your training and validation metrics, or other metric types you'd like to separate:

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5, 
    "val/accuracy": 0.7
}
run.log(metrics)
```

{{< img src="/images/integrations/integrations_add_any_lib_log.png" alt="W&B Workspace" >}}

[Learn more about `run.log`]({{< relref "/guides/models/track/log" >}}).

#### Prevent x-axis misalignments

If you perform multiple calls to `run.log` for the same training step, the wandb SDK increments an internal step counter for each call to `run.log`. This counter may not align with the training step in your training loop.

To avoid this situation, define your x-axis step explicitly with `run.define_metric`, one time, immediately after you call `wandb.init`:

```python
with wandb.init(...) as run:
    run.define_metric("*", step_metric="global_step")
```

The glob pattern, `*`, means that every metric will use `global_step` as the x-axis in your charts. If you only want certain metrics to be logged against `global_step`, you can specify them instead:

```python
run.define_metric("train/loss", step_metric="global_step")
```

Now, log your metrics, your `step` metric, and your `global_step` each time you call `run.log`:

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    run.log({"global_step": step, "train/loss": 0.1})
    run.log({"global_step": step, "eval/loss": 0.2})
```

If you do not have access to the independent step variable, for example "global_step" is not available during your validation loop, the previously logged value for "global_step" is automatically used by wandb. In this case, ensure you log an initial value for the metric so it has been defined when it’s needed.

#### Log images, tables, audio, and more

In addition to metrics, you can log plots, histograms, tables, text, and media such as images, videos, audios, 3D, and more.

Some considerations when logging data include:

* How often should the metric be logged? Should it be optional?
* What type of data could be helpful in visualizing?
  * For images, you can log sample predictions, segmentation masks, etc., to see the evolution over time.
  * For text, you can log tables of sample predictions for later exploration.

[Learn more about logging]({{< relref "/guides/models/track/log" >}}) media, objects, plots, and more.

### Distributed training

For frameworks supporting distributed environments, you can adapt any of the following workflows:

* Detect which is the "main" process and only use `wandb` there. Any required data coming from other processes must be routed to the main process first. (This workflow is encouraged).
* Call `wandb` in every process and auto-group them by giving them all the same unique `group` name.

See [Log Distributed Training Experiments]({{< relref "/guides/models/track/log/distributed-training.md" >}}) for more details.

### Log model checkpoints and more

If your framework uses or produces models or datasets, you can log them for full traceability and have wandb automatically monitor your entire pipeline through W&B Artifacts.

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="Stored Datasets and Model Checkpoints in W&B" >}}

When using Artifacts, it might be useful but not necessary to let your users define:

* The ability to log model checkpoints or datasets (in case you want to make it optional).
* The path/reference of the artifact being used as input, if any. For example, `user/project/artifact`.
* The frequency for logging Artifacts.

#### Log model checkpoints

You can log Model Checkpoints to W&B. It is useful to leverage the unique `wandb` Run ID to name output Model Checkpoints to differentiate them between Runs. You can also add useful metadata. In addition, you can also add aliases to each model as shown below:

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800} 

artifact = wandb.Artifact(
                name=f"model-{run.id}", 
                metadata=metadata, 
                type="model"
                ) 
artifact.add_dir("output_model") # local directory where the model weights are stored

aliases = ["best", "epoch_10"] 
run.log_artifact(artifact, aliases=aliases)
```

For information on how to create a custom alias, see [Create a Custom Alias]({{< relref "/guides/core/artifacts/create-a-custom-alias/" >}}).

You can log output Artifacts at any frequency (for example, every epoch, every 500 steps, and so on) and they are automatically versioned.

#### Log and track pre-trained models or datasets

You can log artifacts that are used as inputs to your training such as pre-trained models or datasets. The following snippet demonstrates how to log an Artifact and add it as an input to the ongoing Run as shown in the graph above.

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
run.use_artifact(artifact_input_data)
```

#### Download an artifact

You re-use an Artifact (dataset, model, etc.) and `wandb` will download a copy locally (and cache it):

```python
artifact = run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts can be found in the Artifacts section of W&B and can be referenced with aliases generated automatically (`latest`, `v2`, `v3`) or manually when logging (`best_accuracy`, etc.).

To download an Artifact without creating a `wandb` run (through `wandb.init`), for example in distributed environments or for simple inference, you can instead reference the artifact with the [wandb API]({{< relref "/ref/python/public-api/index.md" >}}):

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

For more information, see [Download and Use Artifacts]({{< relref "/guides/core/artifacts/download-and-use-an-artifact" >}}).

### Tune hyper-parameters

If your library would like to leverage W&B hyper-parameter tuning, [W&B Sweeps]({{< relref "/guides/models/sweeps/" >}}) can also be added to your library.

### Advanced integrations

You can also see what an advanced W&B integrations look like in the following integrations. Note most integrations will not be as complex as these:

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)