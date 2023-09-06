---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a launch job

A job is a blueprint that contains contextual information about a W&B run it is created from; such as the run's source code, software dependencies, hyperparameters, artifact dataset version, and so forth.

Once you have a launch job, you can add them to a pre-configured launch queue[LINK]. A launch agent will poll that queue and send the job (as a Docker image) to the compute resource that was configured on launch queue.


There are three ways to create a launch job:

* With a Python script
* With a Docker image
* With code Git repository

The following tabs show how to create a job based on each use case. 



## Before you get started
Before you create a job, make sure you confirm that:

<!-- 1. Make your code job-friendly. [LINK] -->
1. A launch queue exists that you have permission to add jobs to it. [LINK]
2. A launch agent is polling the queue you will add launch jobs to. [LINK]



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

<!-- ## Create a launch job -->

<!-- :::info
W&B will check and create jobs based on what requirements you satisfy (if you log a code artifact, associate a run with a git commit, or associate a run with a Docker image).

You can explicitly tell W&B how to create a launch job with wandb.Settings Class. Set the `job_source` parameter to either: `artifact`, `git`, or `image`.
::: -->


### Check queue exists and agent is polling
Find out the name of your queue and the entity it belongs to. Then, follow these instructions to find out the status of your queue and if an agent is polling that queue:

1. Navigate to [wandb.ai/launch](https://wandb.ai/launch).
2. From the **All entities** dropdown, select the entity the launch queue belongs to. 
This will filter queues based on W&B entities.
3. From the filtered results, check that the queue exists.
4. Hover your mouse to the right of the launch queue and select `View queue`.
5. Select the **Agents** tab. Within the **Agents** tab you sill see a list of Agent IDs and their statuses. Ensure that one of the agent IDs has a **polling** status.



## Create a job with a Python script

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="cli">


Create a launch job with with the  W&B CLI. 

Ensure the path with your Python script has a `requirements.txt` file with the Python dependencies required to run your code. A Python runtime is also required. The python runtime can either be specified manually with the runtime parameter or can be auto-detected from a `runtime.txt` or `.python-version file`.

Copy and paste the following code snippet. Replace the values within `"<>"` based on your use case:

```bash
wandb job create --project "<project-name>" -e "<your-entity>" \ 
--name "<name-for-job>" code "<path-to-script/code.py>"
```

For a full list of flags you can use, see the `wandb job create` command documentation. [LINK]

:::note
You do not need to use the `run.log_code()` function within your Python script when you create a launch job with the W&B CLI.
:::


  </TabItem>
  <TabItem value="sdk">

Log your code as an artifact to create a launch job. To do so, log your code to your run as an artifact by calling `run.log_code()`[LINK]. 

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

You can specify a name for your job with the `WANDB_JOB_NAME` environment variable. You can also specify a name by setting the `job_name` parameter in `wandb.Settings` and passing it to `wandb.init`. For example:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

 If you do not specify a name, W&B will automatically generate a launch job name for you. It will create a job name with the following format: `job-<code-artifact-name>`. 



For more information on the `run.log_code()` command, see the API Reference guide. [LINK]

  </TabItem>
</Tabs>




## Create a job with a Docker image
Create a job with a Docker image with the W&B CLI or by creating a Docker container from the image. To create an image-based job, you must first create the Docker image. The Docker image should contain the source code (such as the Dockerfile, requirements.txt file, and so on) required to execute the W&B run. 

As an example, suppose we have a directory called [`fashion_mnist_train`](https://github.com/wandb/launch-jobs/tree/main/jobs/fashion_mnist_train) with the following directory structure:


```
fashion_mnist_train
│   data_loader.py
│   Dockerfile
│   job.py
│   requirements.txt
└───configs
│   │   example.yml
```

We create a Docker image called `fashion-mnist` with the `docker build` command:

```bash
docker build . -t fashion-mnist 
```
For more information on how to build Docker images, see the [Docker build reference documentation](https://docs.docker.com/engine/reference/commandline/build/). 


<Tabs
  defaultValue="cli"
  values={[
    {label: 'W&B CLI', value: 'cli'},
    {label: 'Docker build', value: 'build'},
  ]}>
  <TabItem value="cli">

Create a launch job with the W&B CLI. Copy the following code snippet and replace the values within `"<>"` based on your use case:

```bash
wandb job create --project "<project-name>" --entity "<your-entity>" \ 
--name "<name-for-job>" image image-name:tag
```

For a full list of flags you can use, see the `wandb job create` command documentation. [LINK]

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

You can specify a name for your job with the `WANDB_JOB_NAME` environment variable. If you do not specify a name, W&B will automatically generate a launch job name for you. W&B will create a job name with the following format: `job-<image>-<name>`. 
 
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
--name "<name-for-job>" git https://github.com/repo-name \ 
--entry-point "<path-to-script/code.py>"
```

  </TabItem>
  <TabItem value="git">

Ensure the path with your Python script has a `requirements.txt` file with the Python dependencies required to run your code. A Python runtime is also required. The python runtime can either be specified manually with the runtime parameter or can be auto-detected from a `runtime.txt` or `.python-version file`.


You can specify a name for your job with the `WANDB_JOB_NAME` environment variable. If you do not specify a name, W&B will automatically generate a launch job name for you. It will create a job name with the following format: `job-<git-remote-url>-<path-to-script>`. 

</TabItem>
</Tabs>