---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a launch job

A job is a blueprint that contains contextual information about a W&B run it is created from; such as the run's source code, software dependencies, hyperparameters, artifact dataset version, and so forth.

Once you have one or more jobs, you can add them to a pre-configured launch queue[LINK]. A launch agent will poll that queue and send the job (as a Docker image) to the compute resource that was configured on launch queue.


There are three ways to create a launch job:

* With a Python script
* With a Docker image
* With Git

The following tabs show how to create a job based on each use case. 



## Before you get started
Before you create a job, make sure:

<!-- 1. Make your code job-friendly. [LINK] -->
1. A launch queue exists for you to add jobs to. [LINK]
2. A launch agent is polling the queue you want to add jobs to. [LINK]



<!-- ### Make your code job-friendly
Store your hyperparameters as a dictionary within the `wandb.config` object. Launch jobs are parameterized by the values in your `wandb.config`. This means that When your job is executed (and `wandb.init` is called), your `wandb.config` is populated with the values specified for that run.

In other words, you can change the hyperparameters for future runs created by jobs if you pass your hyperparameters to `wandb.config`.


The following code snippet shows how you can pass hyperparameters to wandb.config when you initialize a W&B run:
```python
epochs=10, 
lr=0.01

run = wandb.init(
    project="<your-project-name>",
    entity='<your-entity>',
    config={
        "learning_rate": lr,
        "epochs": epochs,
    })
```

If you read and store your parameters using a different method, you can still use W&B Launch to run your job, you will need to update your code to use the `wandb.config`.

For example, if you use `argparse` to parse your command line arguments, you can easily adapt your script to use the `wandb.config` to store the values of your arguments. For example:

```python
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# Pass the arguments to wandb.init to store them in
# wandb.config and allow the launch agent to modify them.
wandb.init(config=args)

# Use the values in wandb.config instead of args.
args = wandb.config
``` -->



### Check queue exists and agent is polling

TEXT. 


<!-- ## Create a launch job -->

<!-- :::info
W&B will check and create jobs based on what requirements you satisfy (if you log a code artifact, associate a run with a git commit, or associate a run with a Docker image).

You can explicitly tell W&B how to create a launch job with wandb.Settings Class. Set the `job_source` parameter to either: `artifact`, `git`, or `image`.
::: -->



## Create a job with a Python script

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="cli">

```bash
wandb job create --project "project-name" -e <wandb-entity> \ 
--name "name-for-job" code ./<path-to-python-script>
```

  </TabItem>
  <TabItem value="sdk">

Log your code as an artifact to create a launch job. To do so, log your code to your run as an artifact by calling `run.log_code()`[LINK]. 

The following sample Python code shows how to integrate the `run.log_code()` command (see highlighted portion) into a Python script.

```python title="create_simple_job.py"
import random
import wandb

def run_training_run(epochs, lr):
    settings = wandb.Settings(job_source="artifact")
    run = wandb.init(
        project="launch_demo",
        job_type="eval",
        settings=settings,
        entity='<your-entity>',
        # Simulate tracking hyperparameters
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
        wandb.log({"acc": acc, "loss": loss})

    # highlight-next-line
    run.log_code()
    run.finish()

run_training_run(epochs=10, lr=0.01)
```

For more information on the `run.log_code()` command, see the API Reference guide. [LINK]

  </TabItem>
</Tabs>




## Create a job with a Docker image
Create an image-based job with Docker. 


<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Autogenerate from Docker image', value: 'orange'},
  ]}>
  <TabItem value="cli">Text.</TabItem>
  <TabItem value="orange">Text.</TabItem>
</Tabs>



## Create a job with Git
Text.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Autogenerate from git commit', value: 'git'},
  ]}>
  <TabItem value="cli">

```bash
wandb job create --project "project-name" --entity "your-entity" \ 
--name "name-for-job" git https://github.com/repo-name \ 
--entry-point 'path-to-script/code.py'
```

  </TabItem>
  <TabItem value="git">Text.</TabItem>
</Tabs>