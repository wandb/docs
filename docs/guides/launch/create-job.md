---
description: Discover how to create a job for W&B Launch.
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a job

A job is a complete blueprint of how to perform a step in your ML workflow, like training a model, running an evaluation, or deploying a model to an inference server. For more information, see the [details of launched jobs section](launch-jobs#view-details-of-launched-jobs).

:::danger
`wandb>=0.13.8` is required in order to create jobs.
:::

## How do I create a job?

Jobs will be **captured automatically** from any runs that you track with W&B if you also track the runs source code. You can track source code for your runs in the following ways:

<Tabs
  defaultValue="artifact"
  values={[
    {label: 'Code artifact', value: 'artifact'},
    {label: 'GitHub', value: 'github'},
    {label: 'Docker image', value: 'docker'},
  ]}>

<TabItem value="artifact">

Log code to your run as an Artifact by calling `run.log_code()`.
Example:

```python
run = wandb.init()
run.log_code(".", include_fn=lambda path: path.endswith(".py"))
```

</TabItem>

<TabItem value="github">

Associate your run with a git commit from your GitHub repo. W&B will automatically capture a commit id and diff when `wandb.init()` is called by code in a git repository. If your repository contains a `requirements.txt` file, the launch agent will install any specified dependencies when running your job.

</TabItem>

<TabItem value="docker">

Associate your run with a Docker image. W&B will look for an image tag in the `WANDB_DOCKER` environment variable when `wandb.init()` is called. If `WANDB_DOCKER` is set, a job will be created from the specified image tag. When you launch a job from a Docker image, the launch agent run the image specified.

For more information on `WANDB_DOCKER` see the docs for our [docker integration](/docs/guides/integrations/other/docker.md).

Be sure to set the `WANDB_DOCKER` environment variable to the full image tag as you it will be accessible to your launch agent. For example, if your agent will run images from an ECR repository, you should set `WANDB_DOCKER` to the full image tag, including the ECR repository URL, e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest`.

</TabItem>

</Tabs>

## Job naming conventions

By default, W&B automatically creates a job name for you. The name is generated depending on how the job is created (GitHub, code artifact, or Docker image). The following table describes the job naming convention used for each job source:

| Source        | Naming convention                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |

## Making your code job-friendly

Jobs are parameterized by the values in your `wandb.config`. When your job is executed and `wandb.init` is called, your `wandb.config` will be populated with the values specified for this run. This means it's important that you use the `wandb.config` to store and access all of the parameters that you want to be able to vary between runs.

If you read and store your parameters using a different method, you can still use W&B Launch to run your job, you will need to update your code to use the `wandb.config`.

For example, if you use `argparse` to parse your command line arguments, you can easily adapt your script to use the `wandb.config` to store the values of your arguments. For example:

```python
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# Pass the arguments to wandb.init to store them in wandb.config and allow
# the launch agent to modify them. 
wandb.init(config=args)

# Use the values in wandb.config instead of args. The wandb.config supports
# dot and dictionary style access, e.g. wandb.config.learning_rate or 
# wandb.config["learning_rate"].
args = wandb.config
```
