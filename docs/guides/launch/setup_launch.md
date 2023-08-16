# Set up Launch

The following page describes the general steps to set up W&B Launch: 
1. Create a queue
2. Configuring an agent
3. Deploy an agent. 

For detailed information on how to complete these steps based on your compute resource, see the associated pages:

* Set up Launch for Docker[LINK]
* Set up Launch for SageMaker[LINK]
* Set up Launch for Kubernetes[LINK]

## Create a queue
1. Navigate to Launch App at [wandb.ai/launch](https://wandb.ai/launch). 
2. Click the **create queue** button on the top right of the screen. A drawer will appear. Based on your use case, provide the following information:

* **Entity**: the owner of the queue. The entity can either be your W&B account or any W&B team you are a member of.
* **Queue name**: the name of the queue. 
* **Resource**: the compute resource type for jobs in this queue.
* **Configuration**: default resource configurations for all jobs placed on this queue. Used to configure the compute resources that any job in this queue will run on. These defaults can be overridden when you enqueue a job.

![](/images/launch/create-queue.gif)

3. Select **Create queue**.

## Configure a queue
The schema of your launch queue configuration depends on the compute resource you will use for your jobs. In general, the queue configuration will appear similar to the following:

[INSERT]




By queue type, the rough nature of each queue is:

| Queue type | Configuration | Additional docs |
|------------|---------------|-----------------|
| Docker     | Named arguments to `docker run`. | [Link](./docker.md#docker-queues) |
| SageMaker  | Partial contents of [SageMaker API's `CreateTrainingJob` request.](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) | [Link](./sagemaker.md#queue-configuration)
| Kubernetes | Kubernetes job spec or custom object. | [Link](./kubernetes.md#queue-configuration)

### Optional - Dynamically configure queue
Queue configs can be dynamically configured using macros that are evaluated when the agent launches a job from the queue.  You can set the following macros:


| Macro             | Description                                           |
|-------------------|-------------------------------------------------------|
| `${project_name}` | The name of the project the run is being launched to. |
| `${entity_name}`  | The owner of the project the run being launched to.   |
| `${run_id}`       | The id of the run being launched.                     |
| `${run_name}`     | The name of the run that is launching.                |
| `${image_uri}`    | The URI of the container image for this run.          |


:::info
Any custom macro, such as `${MY_ENV_VAR}`, is substituted with an environment variable from the agent's environment.
:::


## Start an agent