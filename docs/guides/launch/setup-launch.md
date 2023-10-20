---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up Launch

This page describes the high-level steps required to set up W&B Launch:

1. **Set up a queue**: Queues are FIFO and possess a queue configuration. A queue's configuration controls where and how jobs are are executed on a target resource.
2. **Set up an agent**: Agents run on your machine/infrastructure and poll one or more queues for launch jobs. When a job is pulled, the agent ensures that the image is built and available. The agent then submits the job to the target resource.


## Set up a queue
Launch queues must be configured to point to a specific target resource along with any additional configuration specific to that resource. For example, a launch queue that points to a Kubernetes cluster might include environment variables or set a custom namespace its launch queue configuration. When you create a queue, you will specify both the target resource you want to use and the configuration for that resource to use.

When an agent receives a job from a queue, it also receives the queue configuration.  When the agent submits the job to the target resource, it includes the queue configuration along with any overrides from the job itself. For example, you can use a job configuration to specify the Amazon SageMaker instance type for that job instance only.

### Create a queue
1. Navigate to Launch App at [wandb.ai/launch](https://wandb.ai/launch). 
2. Click the **create queue** button on the top right of the screen. 

![](/images/launch/create-queue.gif)

3. From the **Entity** dropdown menu, select the entity the queue will belong to. 
  :::tip
  If you choose a team entity, all members of the team will be able to send jobs to this queue. If you choose a personal entity (associated with a username), W&B will create a private queue that only that user can use.
  :::
4. Provide a name for your queue in the **Queue** field. 
5. From the **Resource** dropdown, select the compute resource you want jobs added to this queue to use.
6. Provide a resource configuration in either JSON or YAML format in the **Configuration** field. The structure and semantics of your configuration document will depend on the resource type that the queue is pointing to. For more details, see the dedicated set up page for your target resource.




## Set up a launch agent
Launch agents are long running processes that poll one or more launch queues for jobs. Launch agents dequeue jobs in first in, first out (FIFO) order. When an agent dequeues a job from a queue, it potentially builds an image for that job. The agent then submits the job to the target resource along with configuration options specified in the queue configuration.

<!-- Future: Insert image -->

:::info
Agents are highly flexible and can be configured to support a wide variety of use cases.  The required configuration for your agent(s) will depend on your specific use case. See the dedicated page for [Docker](./setup-launch-docker.md), [Amazon SageMaker](./setup-launch-sagemaker.md), [Kubernetes](./setup-launch-kubernetes.md), or [Vertex AI](./setup-vertex.md).
:::


### Agent configuration
Configure the launch agent with a configuration (config) YAML file named `launch-config.yaml`. By default, W&B will check for the config file in `~/.config/wandb/launch-config.yaml`. You can optionally specify a different directory when you activate the launch agent.

The contents of your launch agent's configuration file will depend on your launch agent's environment, the launch queue's target resource, Docker builder requirements, cloud registry requirements, and so forth. 

Independent of your use case, there are core configurable options for the launch agent:
* `max_jobs`: maximum number of jobs the agent can execute in parallel 
* `entity`: the entity that the queue belongs to
* `queues`: the name of one or more queues for the agent to watch

:::tip
You can use the W&B CLI to specify universal configurable options for the launch agent (instead of the config YAML file): maximum number of jobs, W&B entity, and launch queues. See the [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) command for more information.
:::


The following YAML snippet shows how to specify core launch agent config keys:

```yaml title="launch-config.yaml"
# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

entity: <entity-name>

# List of queues to poll.
queues:
  - <queue-name>
```

### Configure a container builder
The launch agent can be configured to build images. You must configure the agent to use a container builder if you intend to use launch jobs created from git repos or code artifacts. See the [Create a launch job](./create-launch-job.md) for more information on how to create a launch job. 

W&B Launch supports two builders:

* Docker: The Docker builder uses a local Docker daemon to build images.
* [Kaniko](https://github.com/GoogleContainerTools/kaniko):  Kaniko is a Google project that enables image building in environments where a Docker daemon is unavailable. 

:::tip
Use the Kaniko builder if your agent is polling in an environment where a Docker daemon is unavailable (for example, a Kubernetes cluster).

See the [Set up Kubernetes](./setup-launch-kubernetes.md) for details about the Kaniko builder.
:::

To specify an image builder, include the builder key in your agent configuration. For example, the following code snippet shows a portion of the launch config (`launch-config.yaml`) that specifies to use Docker or Kaniko:

```yaml title="launch-config.yaml"
builder:
	type: docker | kaniko
```

### Configure a cloud registry
In some cases, you might want to connect a launch agent to a cloud registry. Common scenarios where you might want to connect a launch agent to a cloud registry include:

* You want to use the agent to build images and run these images on Amazon SageMaker or VertexAI.
* You want the launch agent to provide credentials to pull from an image repository.

To learn more about how to configure the agent to interact with a cloud registry, see the [Advanced agent set](./setup-agent-advanced.md) up page.

## Activate the launch agent
Activate the launch agent with the `launch-agent` W&B CLI command:

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

In some use cases, you might want to have a launch agent polling queues from within a Kubernetes cluster. See the [Advanced queue set up page](./setup-queue-advanced.md) for more information. 


