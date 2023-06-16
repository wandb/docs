---
description: Getting started guide for W&B Launch.
displayed_sidebar: default
---

# Launch on Docker

Learn how to use W&B Launch on Docker.

:::info
For this example, we show you how to use Docker with Launch using this job: https://github.com/wandb/launch-jobs/tree/main/jobs/fashion_mnist_train. Clone the repo to follow along.

The `fashion_minst_train` directory contains all the files you need to create an image based job and to use Launch.
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

For more information, see the [Docker build](https://docs.docker.com/engine/reference/commandline/build/) documentation.
## 2. Create an image-based job
Use `wandb launch` to create an image-based job. Following the example above, we specify the `fashion_mnist_train` image:


```bash
wandb launch -d fashion_mnist_train
```

Copy and past the hyperlink returned from job to view the run.  Or navigate to your project's workspace.

## 3. Create a queue
Follow the queue creation steps on [this page](../launch/create-queue.md) and select Docker as the resource type.

![](/images/launch/create-queue.gif)


### Optional queue configurations for Docker
Accepted configuration values include all arguments available for the `docker run` command. For more information, see the [reference](https://docs.docker.com/engine/reference/commandline/run). To handle options that can be specified more than once, place the values in a list:

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

:::tip
Add the following resource configuration in order to use GPUs in jobs submitted to this queue:

```json
{
    "gpus": "all"
}
```

The `gpus` key of the resource configuration is used to pass values to the `--gpus` argument of `docker run`. This argument can be used to control which GPUs will be used for by a launch agent when it picks up runs from this queue. For more information, see the relevant [NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration).

Reminder: you will also need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the machine where your agent is running in order to leverage GPU through Docker.
:::


<!-- TODO: put this in a technical FAQ or in the queue docs -->
For jobs that use tensorflow on GPU, you may also need to specify a custom base image for the container build that the agent will perform in order for your runs to properly utilize GPUs. This can be done by adding an image tag under the `builder.cuda.base_image` key to the resource configuration. For example:

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


## 4. Add image-based job to your queue
1. Select the **Jobs** icon on the left panel. 
2. On the right panel, select the name of the queue you created in the previous step. 
3. Select **Launch now**.


## 5. Start your agent
1. Go to the Launch App. 
2. Select the entity from the dropdown that contains your queue.
3. Select your queue from the list. 
4. Select **View queue**.
5. If you do not have an active agent, select **Add an agent**. 
6. Copy and paste the command to the machine that created the Docker image. It will look similar to:

```bash
wandb launch-agent -e <entity> -q <queue-name>
```
See the [Start an agent](./run-agent.md) for more information on agents work and optional configuration.

## Launch agent setup
The agent's default behavior is to perform any necessary container builds by running `docker build` on its local host. The agent will execute runs from a Docker queue by running `docker run` on its local host. The uses the Docker CLI for these actions, so any Docker CLI configuration that you have set up on your machine will be used by the agent.

