---
description: Getting started guide for W&B Launch.
displayed_sidebar: default
---

# Launch on Docker

Learn how to use W&B Launch on Docker.

:::info
We refer to this example: https://github.com/wandb/launch-jobs/tree/main/jobs/fashion_mnist_train for demonstrative purposes in this guide. You can optionally clone the repo if you want to follow along. Otherwise, replace the values with your own.
:::
## Requirements

Before you get started, make sure you satisfy the following:
1. Ensure the Docker daemon is running on the machine that will run your Docker container and W&B Launch agent.
2. Ensure Docker CLI is installed wherever you run your [agent](run-agent.md). 
3. Obtain your [W&B API key](https://wandb.ai/authorize).

See the [Docker documentation](https://docs.docker.com/get-docker/) for more information on how to install Docker.

:::note
If you want the agent to make use of GPUs on Docker, you will also need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
:::


## Build your Docker image
Use the [`docker build`](https://docs.docker.com/engine/reference/commandline/build/) command to build your Docker image with the Docker CLI.  


For this example, we build a Docker image called `fashion_mnist_train`:

```bash
docker build -t fashion_mnist_train .
```

For more information on how to build a Docker image, see the official [Docker build](https://docs.docker.com/engine/reference/commandline/build/) documentation.
## Create an image-based job
Next, create an image-based launch job. To do this, [run your Docker container](https://docs.docker.com/engine/reference/commandline/run/). Pass in your [W&B API key](https://wandb.ai/authorize) and your Docker image as environment variables `WANDB_API_KEY` and `WANDB_DOCKER`, respectively.

Continuing the example above, we specify the `fashion_mnist_train` Docker image:

```bash
docker run -e WANDB_API_KEY="<your-wandb-api-key>" \ 
   -e WANDB_DOCKER="fashion_mnist_train"  fashion_mnist_train:latest
```

This will create an image-based W&B Launch Job. 

## Create a queue
Create a queue in the W&B App that uses Docker as its compute resource:

1. Navigate to the [Launch application](https://wandb.ai/launch).
3. Click on the **Create Queue** button.
4. Select the **Entity** you would like to create the queue in.
5. Enter a name for your queue in the **Name** field.
6. Select **Docker** as the **Resource**. 
7. Define your Docker queue configuration in the **Configuration** field. Pass in any arguments available for the [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command. 

:::tip
Leave the **Configuration** field blank (see .gif below) if you do not need to pass optional [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) parameters.
:::

To handle options that can be specified more than once, place the values in a list within the **Configuration** field:

```json
{
    "env": [
        "MY_ENV_VAR=value",
        "MY_EXISTING_ENV_VAR"
    ],
    "volume": [
         "/mnt/datasets:/mnt/datasets"
    ]
}
```

:::note
Add the following resource configuration to use GPUs:

```json
{
    "gpus": "all"
}
```

The `gpus` key of the resource configuration is used to pass values to the `--gpus` argument of `docker run`. This argument can be used to control which GPUs will be used for by a launch agent when it picks up runs from this queue. For more information, see the relevant [NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration).

Reminder: you will also need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the machine where your agent is running in order to leverage GPU through Docker.
:::

<!-- TODO: put this in a technical FAQ or in the queue docs -->
For jobs that use tensorflow on GPU, you might also need to specify a custom base image for the container build that the agent will perform in order for your runs to properly utilize GPUs. This can be done by adding an image tag under the `builder.cuda.base_image` key to the resource configuration. For example:

```json
{
    "gpus": "all",
    "builder": {
        "cuda": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```



8. After you configure your queue, click on the **Create Queue** button.

![](/images/launch/create-queue.gif)


## Add image-based job to the queue
Follow these steps to add your launch job to your queue:

1. Navigate to your W&B project where the job is defined.
2. Select the **Jobs** icon on the left panel (thunderbolt image). This will redirect you to the Jobs page within your project workspace.
3. Hover your mouse on the right side of the job name and click on the **Launch** button that appears.  A drawer will appear from the right side of your screen. Provide the following:
   * The name your queue from the **Queue** dropdown menu. If you have not created a queue yet, see the [Create a queue section](#3-create-a-queue).
   * Select a W&B project from the **Destination project** dropdown menu. 
4. Click **Launch now**.



## Start your agent
1. Go to the Launch App at [https://wandb.ai/launch](https://wandb.ai/launch). 
2. Select the entity from the **Entity** dropdown that contains your queue.
3. Find your queue and click on **View queue**.
4. Click on **Add an agent**. 
5. A modal will appear. Copy and paste the command to the machine that had the Docker container running. It will look similar to:

```bash
wandb launch-agent -e <entity> -q <queue-name>
```

Replace the values in `<>` with your W&B entity and the name of your queue.

See the [Start an agent](./run-agent.md) for more information on how agents work and for information on optional configuration.

:::note
The agent's default behavior is to perform any necessary container builds by running `docker build` on its local host. The agent will execute runs from a Docker queue by running `docker run` on its local host. The agent uses the Docker CLI for these actions, so any Docker CLI configuration that you have set up on your machine will be used by the agent.
:::
