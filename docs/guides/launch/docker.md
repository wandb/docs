---
description: Getting started guide for W&B Launch.
---

# Launch on Docker

Learn how to use W&B Launch on Docker.

## Launch agent setup

Ensure you [enable the W&B Launch UI](./intro.md) and install Docker on the machine where you will run your launch agent.See the [Docker documentation](https://docs.docker.com/get-docker/) for more information on how to install Docker, and make sure the Docker daemon is running on your machine before you proceed. If you want the agent to make use of GPUs on Docker, you will also need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

The agent's default behavior is to perform any necessary container builds by running `docker build` on its local host. The agent will execute runs from a Docker queue by running `docker run` on its local host. The uses the Docker CLI for these actions, so any Docker CLI configuration that you have set up on your machine will be used by the agent.

## Docker queues

To create a Docker queue, follow the queue creation steps on [this page](../launch/create-queue.md#create-a-queue-1) and select Docker as the resource type.

![](/images/launch/create-queue.gif)

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
:::

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
