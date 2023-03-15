---
description: Prerequisites for W&B Launch.
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Prerequisites
Before you use W&B Launch ensure you have satisfied these prerequisites.

1. Enable the W&B Launch feature flag.
2. Create a queue.
3. Activate a launch agent.


:::info
The steps required to create a queue and activate a launch agent depend on whether or not your queue will use a cloud compute resource (such as Kubernetes on EKS) or use a local Docker container that uses your machine's hardware.
:::

## Enable Launch UI 

Before you get started, ensure you enable the W&B Launch UI and install Docker. See the Docker documentation for more information on how to install Docker. Make sure Docker is running on your local machine.

1. Navigate to your settings page at [https://wandb.ai/settings](https://wandb.ai/settings) .
2. Scroll to the **Beta Features** sections. Toggle the W&B Launch.


Based on your use case, read the following to satisfy W&B Launch prerequisites.


## Set up cloud compute resource
Set up a cloud compute resource and a launch agent to communicate with the cloud resource. To do this, create a queue that is configured to use a cloud resource.


### Create a queue
A queue is where you add W&B Jobs to. Launch Queues possess one or more W&B Jobs that are executed when you start a W&B Launch agent. A Launch Queue can use  any resource configuration that is defined within a given W&B Entity.  Queues give your team members access to specific types of compute resources so they can easily launch jobs at scale. 

1. Navigate to the W&B App at https://wandb.ai/home.
2. Select Launch within the Applications section of the left sidebar.
3. Choose **Create queue**.
4. Select the queue's entity from the **Entity** dropdown menu.
5. Provide a name for the queue in the **Queue name** field
6. Specify a compute resource from the **Resource** dropdown menu. Within the modal, specify the compute configuration in the **Configuration** editor.

![](/images/launch/create_a_queue.png)

W&B currently supports compute configuration for Kubernetes on EKS. For more information on configuration details, contact sales at https://wandb.ai/site/local-contact. 

:::note
You must use Amazon’s Elastic Container Registry (ECR) to use Kubernetes on EKS. Additionally, you will also need an Amazon Simple Storage Service (S3) bucket if you want to build your own Docker container. 
:::

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

![](/images/launch/terminal_gs.png)