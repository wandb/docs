---
description: Launch agent documentation
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Start an agent

Start a launch agent to execute jobs from your queues.

## Overview

A **launch agent** is a long-running process that polls on one or more launch queues and executes the jobs that it pops from the queue. A launch agent can be started with the `wandb launch-agent` command and is capable on launching jobs onto a multitude of compute platforms, including docker, kubernetes, sagemaker, and more. The launch agent is implemented entirely in our [open source client library](https://github.com/wandb/wandb).

## How it works

The agent polls on one or more queues. When the launch agent pops an item from a queue, it will, if necessary, build a container image to execute the run within and then execute that container image on the compute platform targeted by the queue.

These container builds and executions happen asynchronously. To control the max number of concurrent jobs that an agent can perform, pass `-j <num-max-jobs>` to `wandb launch-agent` command or set the `max_jobs` field in the agent config file.

### Compile the job into a container image

If your job contains source code from a git repository or code artifact, the launch agent will build a container image that contains the code and all of the dependencies specified in the job.

If your job was sourced from a container image via the `WANDB_DOCKER` environment variable, then this step is skipped.

Launch currently supports building container images with [Docker](https://docker.com) and [Kaniko](https://github.com/GoogleContainerTools/kaniko). Kaniko should only be used when running the agent in a container, to avoid the security risks associated with running docker-in-docker.

### Execute the run

The launch agent will execute the run in the container image that was built in the previous step or specified in the job. How the agent executes the run depends on the type of queue the job was in. 

For example, if the job was in a Docker queue, the agent will execute the run locally with the `docker run` command. If the job was in a Kubernetes queue, the agent will execute the run on a k8s cluster as a [k8s Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) via the k8s API.



<!-- 
### Activate a launch agent
A launch agent listens the queue and executes jobs added to the queue. If certain criteria are met, the launch agent builds an image for the environment and sends the job to the desired compute runner. 

The following tabs provide information on how to activate a launch agent with basic, standard control settings and with more control options with a config file.


<Tabs
  defaultValue="basic"
  values={[
    {label: 'Basic', value: 'basic'},
    {label: 'EKS', value: 'eks'},
  ]}>
  <TabItem value="basic">

Activate a launch agent in your terminal with the `wandb launch-agent` command:

```bash
wandb launch-agent -q <queue-name>
```
Provide a name for your queue for the `queue` flag (`-q`). For more information on optional flags, see the [CLI Reference Guide](../../ref/cli/README.md).


  </TabItem>
  <TabItem value="eks">

For more control options a config file can be placed at `~/.config/wandb/launch-config.yaml` (or the path can be specified with the config flag).

```yaml
base_url: https://api.wandb.ai # TODO: set wandb base url
entity: <entity-name>
max_jobs: -1 # TODO: set max concurrent jobs here
queues:
- default # TODO: set queue name here
environment:
  type: aws
  region: us-east-1 # TODO: set aws region here
registry:
  type: ecr
  repository: # TODO: set ecr repository name here
builder:
  type: kaniko
  build-context-store: s3://my-bucket/... # TODO: set your build context store here
```

**Environment Flags and Options**

- `type`: specified the environment type, options: `local` and `aws`.
- `region`: only used with AWS environment, AWS region.

**Registry Flags and Options:**

- `type`: specifies the registry type, options: `local` (default) and `ecr` (ECR requires `aws` environment).
- `repository`: only used with ECR, the name of the underlying repository.

**Builder Flags and Options:**

- `type` : specifies type: options: `docker`, `noop`, `kaniko`
    - defaults to docker builder
    - `noop` provides no build capabilities, used for launching image sourced jobs
    - `kaniko` only used in `EKS`
- `build-context-store`: only used with kaniko (EKS)


  </TabItem>
</Tabs>




## Local testing with Docker
Set up a queue for local testing with Docker. 

:::info
The following section describes how to create a "Starter queue". This is a queue that W&B created for local testing purposes only. This means that you do not need to manually configure the queue. 
:::

The "Starter queue" uses a Docker image to run the job locally on your machine. Ensure you have Docker installed and running. See the [Docker documentation](https://docs.docker.com/get-docker/) for instructions on how to install Docker based on your operating system.


### Create a starter queue 
The following procedure demonstrates how to create a starter queue:

1. Navigate to your W&B Project Page. 
2. Select the **Jobs Tab** from the W&B App Project page.
3. The Jobs Page displays a list of jobs that were created from previously executed W&B Runs. 
:::tip
If you do not see a job you will need to create one first. See the [Create a job](./create-job.md) page for more information.
:::
4. Select the **Launch** button next to the name of the Job name. A modal will appear.
5. Within the modal select **Starter queue** From the **Queue** dropdown.

![](/images/launch/starter_queue.png)

### Activate a launch agent
Launch queues require an active agent. To start an agent for a local process you will need to execute a CLI command in your local machine’s terminal. 

1. Navigate to the W&B App at https://wandb.ai/home.
2. Select **Launch** within the **Applications** section of the left sidebar.
3. Select **View queue** next to the name of your queue.
![](/images/launch/view_queue_button.png)


4. Select the **Add an agent** button at the top of your screen.
![](/images/launch/add_agent_zoom.png)

5. A modal will appear with a W&B CLI command. Copy this and paste this command into your terminal.

![](/images/launch/add_agent_dark_background.png)

Within your terminal, you will see the agent begin to poll for queues. Your terminal should look similar to the following image:

![](/images/launch/terminal_gs.png) -->