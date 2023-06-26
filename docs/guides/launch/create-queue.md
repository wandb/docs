---
description: Discover how to create a queue for W&B Launch.
displayed_sidebar: default
---

# Create a queue

**Launch queues** are first-in, first-out (FIFO) configurable queues you can submit your jobs to. The jobs are then executed by any polling launch agents. Each item in a launch queue consists of a job and settings for the parameters of that job.  

:::tip
Similar to [launch jobs](./create-job.md), a launch queue belongs to a W&B entity. Both the launch queue and the jobs you want to enqueue must belong to the same W&B entity.
:::


Follow this guide to learn how to create, configure, and view launch queues.

## Create a queue

1. Navigate to [wandb.ai/launch](https://wandb.ai/launch). 
2. Click **create queue** button in the top right of the screen to start creating a new launch queue.

When you click the button, a drawer will slide from the right side of your screen and present you with some options for your new queue:

* **Entity**: the owner of the queue. The entity can either be your W&B account or any W&B team you are a member of.
* **Queue name**: the name of the queue. 
* **Resource**: the compute resource type for jobs in this queue.
* **Configuration**: default resource configurations for all jobs placed on this queue. Used to configure the compute resources that any job in this queue will run on. These defaults can be overridden when you enqueue a job.

![](/images/launch/create-queue.gif)

3. Select **Create queue** to create the queue.

## Queue configuration

The schema and meaning of launch queue configuration depends on the queue type. By queue type, the rough nature of each queue is:

| Queue type | Configuration | Additional docs |
|------------|---------------|-----------------|
| Docker     | Named arguments to `docker run`. | [Link](./docker.md#docker-queues) |
| SageMaker  | Partial contents of [SageMaker API's `CreateTrainingJob` request.](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) | [Link](./sagemaker.md#queue-configuration)
| Kubernetes | Kubernetes job spec or custom object. | [Link](./kubernetes.md#queue-configuration)

While the format of configuration is different for each queue, there are some shared properties. Queue configs can be dynamically configured using macros that will be evaluated when the agent actually launches a job from the queue. The list of available macros at this time is:

| Macro             | Description                                           |
|-------------------|-------------------------------------------------------|
| `${project_name}` | The name of the project the run is being launched to. |
| `${entity_name}`  | The owner of the project the run being launched to.   |
| `${run_id}`       | The id of the run being launched.                     |
| `${run_name}`     | The name of the run that is launching.                |
| `${image_uri}`    | The URI of the container image for this run.          |

Additionally, any custom macro, e.g. `${MY_ENV_VAR}` will be substituted with an environment variable from the agent's environment, if present.

## View queues

View queues associated with a W&B entity with the W&B App.

1. Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home).
2. Select **Launch** within the **Applications** section of the left sidebar.
3. Select the **All entities** dropdown and select the entity to filter with.

A list of queues will appear in the Launch workspace. From left to right, each queue listed shows the name of the queue, the entity the queue belongs to, the compute resource configured for that queue and the state of the queue.

For example, in the following image, we have a queue called **Starter queue**. It belongs the the wandb entity. The queue is configured to use Docker. And the queue is in an **inactive (Not running)** state.

![](/images/launch/launch_queues_all.png)

A queue is either **Active** or **Not running** (inactive).

* **Active**: A queue is active if at least one agent is either polling or executing jobs for that queue.
* **Not running**: A queue is not running if there is no agent polling or executing jobs for that queue.
