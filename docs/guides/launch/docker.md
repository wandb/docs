---
description: Getting started guide for W&B Launch.
displayed_sidebar: default
---

# Launch on Docker

Learn how to use W&B Launch on Docker.

:::info
For this guide, we refer to this repo and directory: https://github.com/wandb/launch-jobs/tree/main/jobs/fashion_mnist_train. The `fashion_minst_train` directory contains all the files you need to create an image based job such as the training script and Dockerfile. Optionally clone the repo if you want to follow along.
:::
## Requirements
Ensure Docker CLI is installed wherever you run your [agent](run-agent.md). Additionally, make sure the Docker daemon is running on your machine before you proceed. 

See the [Docker documentation](https://docs.docker.com/get-docker/) for more information on how to install Docker.

:::note
If you want the agent to make use of GPUs on Docker, you will also need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
:::


## 1. Build your Docker image
Use the `docker build` command to build your Docker image with the Docker CLI.  For this example, we build a Docker image called `fashion_mnist_train`:

```bash
docker build -t fashion_mnist_train .
```

For more information on how to build a Docker image, see the official [Docker build](https://docs.docker.com/engine/reference/commandline/build/) documentation.
## 2. Create an image-based job
Next, create an image-based launch job. To do this, [run your Docker container](https://docs.docker.com/engine/reference/commandline/run/). Pass in your W&B API key and you Docker image as environment variables `WANDB_API_KEY` and `WANDB_DOCKER`, respectively.

Continuing the example above, we specify the `fashion_mnist_train` Docker image:

```bash
docker run -e WANDB_API_KEY="<your-wandb-api-key>" \ 
   -e WANDB_DOCKER="fashion_mnist_train"  fashion_mnist_train:latest
```

This will create an image-based W&B Launch Job. 

## 3. Create a queue
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


## 4. Add image-based job to the queue
1. Navigate to your W&B project where the job is defined.
2. Select the **Jobs** icon on the left panel (thunderbolt image).
3. This will redirect you to the Jobs page within your project workspace.
4. Hover your mouse on the right side of the job name. 
5. Select **Launch**.  A drawer will appear from the right side of your screen. Select the
   * The name of the queue you want to add the job to from the **Queue** dropdown menu.
   * From the **Destination project** dropdown menu, select the W&B project to send the run to.
6. Select **Launch now**.



## 5. Start your agent
1. Go to the Launch App at [https://wandb.ai/launch](https://wandb.ai/launch). 
2. Select the entity from the **Entity** dropdown that contains your queue.
3. Find your queue and click on **View queue**.
4. If you do not have an active agent, select **Add an agent**. 
5. A modal will appear. Copy and paste the command to the machine that had the Docker container running. It will look similar to:

```bash
wandb launch-agent -e <entity> -q <queue-name>
```

Replace the values in `<>` with your W&B entity and the name of your queue.

See the [Start an agent](./run-agent.md) for more information on how agents work and for information on optional configuration.

:::note
The agent's default behavior is to perform any necessary container builds by running `docker build` on its local host. The agent will execute runs from a Docker queue by running `docker run` on its local host. The agent uses the Docker CLI for these actions, so any Docker CLI configuration that you have set up on your machine will be used by the agent.
:::
