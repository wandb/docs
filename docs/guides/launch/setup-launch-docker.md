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

When you use Docker with W&B Launch, W&B will first build an image (optionally, see INSERT for more information), and then build and run a container from that image. The image is built with Docker's `docker run <image-uri>` command. The queue configuration is interpreted as additional arguments that are passed to the `docker run` command.

<!-- Future: Insert diagram -->

## Configure a Docker queue

<!-- The launch queue configuration for a Docker target compute resource accepts the same options defined for the [docker run command](https://www.notion.so/Set-up-for-Docker-e784819393af47e3bba43c648abc67cb?pvs=21). W&B Launch will take the launch queue's configuration you define and reformat it execute the `docker run` command. There are two transformations that take place: -->

The launch queue configuration (for a Docker target resource) accepts the same options defined in Docker's `docker run` CLI command[LINK].

The agent receives options defined in the queue configuration. The agent then merges the received options with any overrides from the launch job’s configuration to produce a final `docker run` command that is executed on the target resource (in this case, a local machine).

There are two syntax transformations that take place:

1. Repeated options are defined in the queue configuration as a list.
2. Flag options are defined in the queue configuration as booleans with the value `true`.

For example, the following queue configuration:

```json 
{
    "env": [
        "MY_ENV_VAR=value",
        "MY_EXISTING_ENV_VAR"
    ],
    "volume": "/mnt/datasets:/mnt/datasets",
		"rm": True
}
```

Results in the following `docker run` command:

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri>
```

:::note
* Volumes can be specified either as a list of strings, or a single string. Use a list if you specify multiple volumes.

* Docker automatically passes environment variables, that are not assigned a value, through from the launch agent environment. This means that, if the launch agent has an env var `MY_EXISTING_ENV_VAR`, that that environment variable is available in the container. This is useful if you want to use other config keys without publishing them in the queue configuration.
:::

## Create a queue

<!-- <Tabs
  defaultValue="cli"
  values={[
    {label: 'W&B cli', value: 'cli'},
    {label: 'Python SDK', value: 'config'},
  ]}>
  <TabItem value="cli">


1. Navigate to the [Launch page](https://wandb.ai/launch).
2. Click on the **Create Queue** button.
3. Select the **Entity** you would like to create the queue in.
4. Enter a name for your queue in the **Name** field.
5. Select **Docker** as the **Resource**.
6. Define your Docker queue configuration in the **Configuration** field.
7. Click on the **Create Queue** button to create the queue.


  </TabItem>
  <TabItem value="config">This is an orange 🍊</TabItem>
</Tabs> -->
Create a queue that uses Docker as compute resource with the W&B cli:

1. Navigate to the [Launch page](https://wandb.ai/launch).
2. Click on the **Create Queue** button.
3. Select the **Entity** you would like to create the queue in.
4. Enter a name for your queue in the **Name** field.
5. Select **Docker** as the **Resource**.
6. Define your Docker queue configuration in the **Configuration** field.
7. Click on the **Create Queue** button to create the queue.


## Configure a launch agent on a local machine 

Configure the launch agent with a YAML config file named `launch-config.yaml`. By default, W&B will check for the config file in `~/.config/wandb/launch-config.yaml`. You can optionally specify a different directory when you activate the launch agent[LINK].

:::tip
You can use the W&B CLI to specify core configurable options for the launch agent (instead of the config YAML file): maximum number of jobs, W&B entity, and launch queues. See the `wandb launch-agent` command for more information. [LINK]
:::

## Core agent config options

The following tabs demonstrate how to specify the core config agent options with the W&B CLI and with a YAML config file:

<Tabs
  defaultValue="cli"
  values={[
    {label: 'W&B CLI', value: 'cli'},
    {label: 'Config file', value: 'config'},
  ]}>
  <TabItem value="cli">

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
* Share built images with others
* Limit the amount of data stored on your local machine 

To learn more about how connect the launch agent with a cloud registry, see Advanced agent set up. [LINK]

<!-- 
The following sections outline how to configure a launch queue and agent to execute jobs in Docker on a local machine. 


## Configure a queue for Docker

Create and configure a queue that uses Docker as its compute resource. Docker queue configurations are used to populate `docker run` arguments. Therefore, you can specify any arguments available for the [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command in your queue configuration.

1. Navigate to the [Launch page](https://wandb.ai/launch).
3. Click on the **Create Queue** button.
4. Select the **Entity** you would like to create the queue in.
5. Enter a name for your queue in the **Name** field.
6. Select **Docker** as the **Resource**. 
7. Define your Docker queue configuration in the **Configuration** field.  
    Place the values in a list within the **Configuration** field to handle options that can be specified more than once:
    For example, the following queue configuration specifies a set of environment variables(`"env"` key) and to bind mount a volume (`"volume"` key):
    ```json title="Queue configuration"
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
7. After you configure your queue, click on the **Create Queue** button to create the queue.

![](/images/launch/create-queue.gif)


## Configure a launch agent for Docker
Create and configure a YAML configuration file for the launch agent. At a minimum, you must specify your W&B entity, the maximum number of jobs, and the name of the queue to use for the `entity`, `max_jobs`, and `queues` keys, respectively.

Copy and paste the code block below. Replace the values based on your use case:


```yaml title="~/.config/wandb/launch-config.yaml"
# W&B entity (i.e. user or team) name
entity: entity-name

# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

# List of queues to poll.
queues:
  - default
```

You can also specify a specific environment to run the agent on, specify a container registry, and specify a specific Docker builder. 

For more information on optional launch agent configuration options, see the Configure a launch agent page. [LINK]


## Start your agent
1. Go to the Launch page at [https://wandb.ai/launch](https://wandb.ai/launch). 
2. Select the entity from the **Entity** dropdown that contains your queue.
3. Find your queue and click on **View queue**.
4. Click on **Add an agent**. 
5. A modal will cliear. Copy and paste the command to the machine that had the Docker container running. It will look similar to:

```bash
wandb launch-agent -e <entity> -q <queue-name>
```

Replace the values in `<>` with your W&B entity and the name of your queue.

See the [Start an agent](./run-agent.md) for more information on how agents work and for information on optional configuration.

:::info
The launch agent's default behavior is to perform any necessary container builds by running `docker build` on its local host. The agent will execute runs from a Docker queue by running `docker run` on its local host. The agent uses the Docker CLI for these actions, so any Docker CLI configuration that you have set up on your machine will be used by the agent.
:::


## (Optional) Additional queue configurations for Docker

### Specify a GPU for your Docker queue
Add the `gpus` key to your queue resource configuration[LINK] to control which NVIDIA GPU the launch agent uses for executing jobs.

:::note
You will need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the machine where your agent is running in order to leverage GPU through Docker.
:::


For example, the following code snippet shows how to [INSERT]:

```json
{
    "gpus": "all"
}
```

For more information, see the relevant [NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration).

:::info
The `gpus` key of the resource configuration is used to pass values to the `--gpus` argument of `docker run`.
:::



### TensorFlow on GPUs

You might need to specify a custom base image for the container builder for the agent if you have a launch job that will TensorFlow on a GPU. To ensure TensorFlow properly utilizes your GPU, specify a Docker image and its image tag for the `builder.accelerator.base_image` key in the queue resource configuration. 

For example, the following JSON code snippet shows how to specify the  `tensorflow/tensorflow:latest-gpu` base image:

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
``` -->