---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up for Docker

The following guide describes how to configure W&B Launch to utilize Docker on a local machine for both the launch agent environment and the queue's target resource.

Using Docker to execute jobs and as the launch agent's environment on the same local machine is particularly useful if your compute is installed on a machine that does not have a cluster management system (such as Kubernetes).

:::tip
This set up is common for users who perform experiments on their local machine, or that have a remote machine that they SSH in to, to submit launch jobs.
:::

When you use Docker with W&B Launch, W&B will first build an image, and then build and run a container from that image. The image is built with the Docker `docker run <image-uri>` command. The queue configuration is interpreted as additional arguments that are passed to the `docker run` command.

<!-- Future: Insert diagram -->

## Configure a Docker queue

<!-- The launch queue configuration for a Docker target compute resource accepts the same options defined for the [docker run command](https://www.notion.so/Set-up-for-Docker-e784819393af47e3bba43c648abc67cb?pvs=21). W&B Launch will take the launch queue's configuration you define and reformat it execute the `docker run` command. There are two transformations that take place: -->

The launch queue configuration (for a Docker target resource) accepts the same options defined in the [`docker run`](../../ref/cli/wandb-docker-run.md) CLI command.

The agent receives options defined in the queue configuration. The agent then merges the received options with any overrides from the launch job’s configuration to produce a final `docker run` command that is executed on the target resource (in this case, a local machine).

There are two syntax transformations that take place:

1. Repeated options are defined in the queue configuration as a list.
2. Flag options are defined in the queue configuration as a Boolean with the value `true`.

For example, the following queue configuration:

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

Results in the following `docker run` command:

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

:::note

- Volumes can be specified either as a list of strings, or a single string. Use a list if you specify multiple volumes.

- Docker automatically passes environment variables, that are not assigned a value, through from the launch agent environment. This means that, if the launch agent has an environment variable `MY_EXISTING_ENV_VAR`, that environment variable is available in the container. This is useful if you want to use other config keys without publishing them in the queue configuration.

:::

## Create a queue

Create a queue that uses Docker as compute resource with the W&B CLI:

1. Navigate to the [Launch page](https://wandb.ai/launch).
2. Click on the **Create Queue** button.
3. Select the **Entity** you would like to create the queue in.
4. Enter a name for your queue in the **Name** field.
5. Select **Docker** as the **Resource**.
6. Define your Docker queue configuration in the **Configuration** field.
7. Click on the **Create Queue** button to create the queue.

## Configure a launch agent on a local machine

Configure the launch agent with a YAML config file named `launch-config.yaml`. By default, W&B will check for the config file in `~/.config/wandb/launch-config.yaml`. You can optionally specify a different directory when you activate the launch agent.

:::tip
You can use the W&B CLI to specify core configurable options for the launch agent (instead of the config YAML file): maximum number of jobs, W&B entity, and launch queues. See the [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) command for more information.
:::

### Enable Nvidia GPU on Docker

In order to use GPU from within a Docker container, you must install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). To test your toolkit install, try running:

```bash
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

If you see the output from `nvidia-smi`, then your toolkit should be installed correctly!

The `--gpus` flag of the `docker run` command allows you to specify which GPU are available to a container.
Use the `gpus` key in the launch queue configuration to specify the value that will be passed to `--gpus` when the agent runs your job on Docker. See [configure a Docker queue](#configure-a-docker-queue) for more information on customizing `docker run` argument set by the agent.

:::tip
To have each run use all available GPU, add the following assignment in your launch queue configuration:

```json
{
  "gpus": "all"
}
```

If you are building images from a code or artifact-sourced job, you can also override the base image used by the agent to include the NVIDIA Container Toolkit. For example, to set the base image to `tensorflow/tensorflow:latest-gpu` set:

```json
{
  "builder": {
    "accelerator": {
      "base_image": "tensorflow/tensorflow:latest-gpu"
    }
  }
}
```

in your launch queue configuration.

:::

For more information on how to use the `gpus` flag, see the [Docker documentation](https://docs.docker.com/config/containers/resource_constraints/#gpu).

## Core agent config options

The following tabs demonstrate how to specify the core config agent options with the W&B CLI and with a YAML config file:

<Tabs
defaultValue="CLI"
values={[
{label: 'W&B CLI', value: 'CLI'},
{label: 'Config file', value: 'config'},
]}>
<TabItem value="CLI">

```bash
wandb launch-agent -q <queue-name> --max-jobs <n>
```

  </TabItem>
  <TabItem value="config">

```yaml title="launch-config.yaml"
max_jobs: <n concurrent jobs>
queues:
	- <queue-name>
```

  </TabItem>
</Tabs>

## Docker image builders

The launch agent on your machine can be configured to build Docker images. By default, these images are stored on your machine’s local image repository. To enable your launch agent to build Docker images, set the `builder` key in the launch agent config to `docker`:

```yaml title="launch-config.yaml"
builder:
	type: docker
```

## Cloud registries

You might want to connect your launch agent with a cloud registry. Connecting your launch agent to a cloud registry enables you to:

- Share built images with others
- Limit the amount of data stored on your local machine

To learn more about how connect the launch agent with a cloud registry, see the [Advanced agent set](./setup-agent-advanced.md) up page.
