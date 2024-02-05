---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Create a launch job
<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

A job is a blueprint that contains contextual information about a W&B run it is created from; such as the run's source code, software dependencies, hyperparameters, artifact version, and so forth.

Once you have a launch job, you can add them to a pre-configured [launch queue](./launch-terminology.md#launch-queue). The launch agent that was deployed by you or someone on your team, will poll that queue and send the job (as a Docker image) to the compute resource that was configured on launch queue.

There are three ways to create a launch job:

- [With a Python script](#create-a-job-with-a-wb-artifact)
- [With a Docker image](#create-a-job-with-a-docker-image)
- [With Git repository](#create-a-job-with-git)

The following sections show how to create a job based on each use case.


## Before you get started

Before you create a launch job, find out the name of your queue and the entity it belongs to. Then, follow these instructions to find out the status of your queue and to check if an agent is polling that queue:

1. Navigate to [wandb.ai/launch](https://wandb.ai/launch).
2. From the **All entities** dropdown, select the entity the launch queue belongs to.
3. From the filtered results, check that the queue exists.
4. Hover your mouse to the right of the launch queue and select `View queue`.
5. Select the **Agents** tab. Within the **Agents** tab you sill see a list of Agent IDs and their statuses. Ensure that one of the agent IDs has a **polling** status.

## Create a job with a W&B artifact

<Tabs
defaultValue="cli"
values={[
{label: 'CLI', value: 'cli'},
{label: 'Python SDK', value: 'sdk'}
]}>
<TabItem value="cli">

Create a launch job with the W&B CLI.

Ensure the path with your Python script has a `requirements.txt` file with the Python dependencies required to run your code. A Python runtime is also required. The python runtime can either be specified manually with the runtime parameter or can be auto-detected from a `runtime.txt` or `.python-version file`.

Copy and paste the following code snippet. Replace the values within `"<>"` based on your use case:

```bash
wandb job create --project "<project-name>" -e "<your-entity>" \
--name "<name-for-job>" code "<path-to-script/code.py>"
```

For a full list of flags you can use, see the [`wandb job create`](../../ref/cli/wandb-job/wandb-job-create.md) command documentation.

:::note
You do not need to use the [`run.log_code()`](../../ref/python/run.md#log_code) function within your Python script when you create a launch job with the W&B CLI.
:::

  </TabItem>
  <TabItem value="sdk">

Log your code as an artifact to create a launch job. To do so, log your code to your run as an artifact with [`run.log_code()`](../../ref/python/run.md#log_code).

The following sample Python code shows how to integrate the `run.log_code()` function (see highlighted portion) into a Python script.

```python title="create_simple_job.py"
import random
import wandb


def run_training_run(epochs, lr):
    settings = wandb.Settings(job_source="artifact")
    run = wandb.init(
        project="launch_demo",
        job_type="eval",
        settings=settings,
        entity="<your-entity>",
        # Simulate tracking hyperparameters
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # simulating a training run
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        wandb.log({"acc": acc, "loss": loss})

    # highlight-next-line
    run.log_code()
    run.finish()


run_training_run(epochs=10, lr=0.01)
```

You can specify a name for your job with the `WANDB_JOB_NAME` environment variable. You can also specify a name by setting the `job_name` parameter in `wandb.Settings` and passing it to `wandb.init`. For example:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

If you do not specify a name, W&B will automatically generate a launch job name for you. It will create a job name with the following format: `job-<code-artifact-name>`.

For more information on the [`run.log_code()`](../../ref/python/run.md#log_code) command, see the [API Reference guide](../../ref/README.md).

  </TabItem>
</Tabs>

## Create a job with a Docker image

Create a job with a Docker image with the W&B CLI or by creating a Docker container from the image. To create an image-based job, you must first create the Docker image. The Docker image should contain the source code (such as the Dockerfile, requirements.txt file, and so on) required to execute the W&B run.

As an example, suppose you have a directory called [`fashion_mnist_train`](https://github.com/wandb/launch-jobs/tree/main/jobs/fashion_mnist_train) with the following directory structure:

```
fashion_mnist_train
│   data_loader.py
│   Dockerfile
│   job.py
│   requirements.txt
└───configs
│   │   example.yml
```

You can create a Docker image called `fashion-mnist` with the `docker build` command:

```bash
docker build . -t fashion-mnist
```

For more information on how to build Docker images, see the [Docker build reference documentation](https://docs.docker.com/engine/reference/commandline/build/).

<Tabs
defaultValue="cli"
values={[
{label: 'W&B CLI', value: 'cli'},
{label: 'Docker run', value: 'build'},
]}>
<TabItem value="cli">

Create a launch job with the W&B CLI. Copy the following code snippet and replace the values within `"<>"` based on your use case:

```bash
wandb job create --project "<project-name>" --entity "<your-entity>" \
--name "<name-for-job>" image image-name:tag
```

For a full list of flags you can use, see the [`wandb job create`](../../ref/cli/wandb-job/wandb-job-create.md) command documentation.

  </TabItem>
  <TabItem value="build">

Associate your run with a Docker image. W&B will look for an image tag in the `WANDB_DOCKER` environment variable, and if `WANDB_DOCKER` is set, a job will be created from the specified image tag. Ensure that the `WANDB_DOCKER` environment variable is set to the full image tag.

Create a launch job by building a Docker container from a Docker image. Copy the following code snippet and replace the values within `"<>"` based on your use case:

```bash
docker run -e WANDB_PROJECT="<project-name>" \
-e WANDB_ENTITY="<your-entity>" \
-e WANDB_API_KEY="<your-w&B-api-key>" \
-e WANDB_DOCKER="<docker-image-name>" image:tag
```

You can specify a name for your job with the `WANDB_JOB_NAME` environment variable. W&B automatically generates a launch job name for you if you do not specify a name. W&B assigns a job name with the following format: `job-<image>-<name>`.

:::tip
Make sure you specify is set to the full image tag. For example, if your agent will run images from an ECR repository, you should set `WANDB_DOCKER` to the full image tag, including the ECR repository URL: `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:develop`. The docker tag, in this case `'develop'`, is added as an alias to the resulting job.
:::

  </TabItem>
</Tabs>

## Create a job with Git

Create a Git-based job with W&B Launch. Code and other assets are cloned from a certain commit, branch, or tag in a git repository.

<Tabs
defaultValue="cli"
values={[
{label: 'CLI', value: 'cli'},
{label: 'Autogenerate from git commit', value: 'git'},
]}>
<TabItem value="cli">

```bash
wandb job create --project "<project-name>" --entity "<your-entity>" \ 
--name "<name-for-job>" git https://github.com/org-name/repo-name.git \ 
--entry-point "<path-to-script/code.py>"
```

To build from a branch or commit hash, append the `-g` argument.

  </TabItem>
  <TabItem value="git">

Ensure the path with your Python script has a `requirements.txt` file with the Python dependencies required to run your code. A Python runtime is also required. The python runtime can either be specified manually with the runtime parameter or can be auto-detected from a `runtime.txt` or `.python-version file`.

You can specify a name for your job with the `WANDB_JOB_NAME` environment variable. If you do not specify a name, W&B automatically generate a launch job name for you. In this case, W&B assigns a job name with the following format: `job-<git-remote-url>-<path-to-script>`.

</TabItem>
</Tabs>

### Git remote URL handling

The Git remote associated with a launch job can be either an HTTPS or an SSH URL. Git remote URLs typically use the following formats:

- `https://github.com/organization/repository.git` (HTTPS)
- `git@github.com:organization/repository.git` (SSH)

The exact format varies by git hosting provider.

The remote URL format is important because it determines how the git remote is accessed and authenticated. The following table describes requirements you must satisfy to access and authentication:

| Remote URL | Requirements for access and authentication |
| ---------- | ------------------------------------------ |
| HTTPS URL  | username and password to authenticate with the git remote |
| SSH URL    | SSH key to authenticate with the git remote |


The Git remote URL is automatically inferred from the local git repository if your launch job is created automatically by a W&B run. 

If you create a job manually, you are responsible for providing the URL in the desired format. For more information, see [LINK].

## Launch job names

By default, W&B automatically generates a job name for you. The name is generated depending on how the job is created (GitHub, code artifact, or Docker image). Alternatively, you can define a launch job's name with environment variables or with the W&B Python SDK.

### Default launch job names

The following table describes the job naming convention used by default based on job source:

| Source        | Naming convention                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |

### Name your launch job

Name your job with a W&B environment variable or with the W&B Python SDK

<Tabs
defaultValue="env_var"
values={[
{label: 'Environment variable', value: 'env_var'},
{label: 'W&B Python SDK', value: 'python_sdk'},
]}>
<TabItem value="env_var">

Set the `WANDB_JOB_NAME` environment variable to your preferred job name. For example:

```bash
WANDB_JOB_NAME=awesome-job-name
```

  </TabItem>
  <TabItem value="python_sdk">

Define the name of your job with `wandb.Settings`. Then pass this object when you initialize W&B with `wandb.init`. For example:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

  </TabItem>
</Tabs>

:::note
For docker image jobs, the version alias is automatically added as an alias to the job.
:::
