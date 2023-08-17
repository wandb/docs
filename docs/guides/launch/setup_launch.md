---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up Launch

The following page describes the general steps required to set up W&B Launch: 
1. Create a queue
2. Configure a queue
3. Configure an agent. 

:::info
For detailed information on how to complete these steps based on your compute resource, see the associated pages:

* Set up Launch for Docker[LINK]
* Set up Launch for SageMaker[LINK]
* Set up Launch for Kubernetes[LINK]
:::


## Create a queue
1. Navigate to Launch App at [wandb.ai/launch](https://wandb.ai/launch). 
2. Click the **create queue** button on the top right of the screen. 

![](/images/launch/create-queue.gif)

3. From the **Entity** dropdown menu, select the entity the queue will belong to. 
4. Provide a name for your queue in the **Queue** field. 
5. From the **Resource** dropdown, select the compute resource you want jobs added to this queue to use.
6. Provide a resource configuration in either JSON or YAML format in the **Configuration** field.  The next section, [Configure a queue](#configure-a-queue), gives a high level overview of how to configure queues based on the compute resource type you select in the previous step.

:::tip
Jobs place on a queue will use the default configuration specified when you created the queue.  Defaults can be overridden when you enqueue a job. 
:::

## Configure a queue
The schema of your launch queue configuration depends on the compute resource you will use for your jobs. You can provide your configuration in JSON or YAML format. 


The following tabs show example configurations for SageMaker and Kubernetes:

<Tabs
  defaultValue="sagemaker"
  values={[
    {label: 'Sagemaker', value: 'sagemaker'},
    {label: 'Kubernetes', value: 'kubernetes'},
  ]}>
  <TabItem value="sagemaker">

```json title='Queue configuration in W&B App UI'
{
  "RoleArn": "<arn:partition:service:region:account:resource>",
  "ResourceConfig": {
    "InstanceType": "ml.m4.xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 2
  },
  "OutputDataConfig": {
    "S3OutputPath": "https://s3.region-code.amazonaws.com/bucket-name/key-name"
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 3600
  }
}
```

  </TabItem>
  <TabItem value="kubernetes">

```yaml title='Queue configuration in W&B App UI'
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 60
  template:
    spec:
      restartPolicy: Never
```

  </TabItem>
</Tabs>

<!-- Uncomment this when you fix the other pages. -->

<!-- 
The rough nature of each queue is:

| Queue type | Configuration | Additional docs |
|------------|---------------|-----------------|
| Docker     | Named arguments to `docker run`. | [Link](./docker.md#docker-queues) |
| SageMaker  | Partial contents of [SageMaker API's `CreateTrainingJob` request.](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) | [Link](./sagemaker.md#queue-configuration)
| Kubernetes | Kubernetes job spec or custom object. | [Link](./kubernetes.md#queue-configuration) -->

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


## Configure an agent
W&B Launch uses a launch agent to poll one or more launch queues for jobs. The launch agent will pop jobs from the queue (FIFO) and execute them based on the agent configuration file you define in a YAML file called `launch-config.yaml`. By default, the agent configuration file is stored in `~/.config/wandb/`. 

When the launch agent pops a job from the queue, it will build a container image for that job. How the agent builds the container depends on the type of queue the job was in. 

[INSERT IMAGE]

For example, if the job was in a Docker queue, the agent will execute the run locally with the `docker run` command. If the job was in a Kubernetes queue, the agent will execute the run on a Kubernetes cluster as a [Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) with the Kubernetes API.



The contents of your agent configuration file (`launch-config.yaml`) will depend on the compute resource that is targeted by the launch queue. However, you must specify your W&B entity, the maximum number of jobs, and the name of the queue to use for the `entity`, `max_jobs`, and `queues` keys, respectively.

The following YAML snippet lists the required keys you must specify for your agent configuration:

```yaml title="~/.config/wandb/launch-config.yaml"
# W&B entity (i.e. user or team) name
entity: entity-name

# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

# List of queues to poll.
queues:
  - default
```


You can also specify an environment, registry, and [builder](#builders) in your agent configuration. 

:::tip
The following sections describe optional agent configuration options for specifying cloud environments, a specific Docker image builder, and container registries. 
:::

<!-- 
For example: 


```yaml title="~/.config/wandb/launch-config.yaml"
# W&B entity (i.e. user or team) name
entity: <entity-name>

# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

# List of queues to poll.
queues:
  - default

# Cloud environment config.
environment:
  type: aws|gcp|azure

# Container registry config.
registry:
  type: ecr|gcr|acr

# Container build config.
builder:
  type: docker|kaniko|noop
``` -->

### Container builder options

The `builder` key is used to specify the container builder the agent will use to build container images. 

:::info
The `builder` key is not required and if omitted, the agent will use Docker to build images locally.
:::

<Tabs
defaultValue="docker"
values={[
{label: 'Docker', value: 'docker'},
{label: 'Kaniko', value: 'kaniko'},
{label: 'Noop', value: 'noop'}
]}>

<TabItem value="docker">

Set the builder type to `docker` to use Docker to build images locally. 

```yaml title="~/.config/wandb/launch-config.yaml"
# W&B entity (i.e. user or team) name
entity: entity-name

# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

# List of queues to poll.
queues:
  - default

// highlight-start
builder:
  type: docker
// highlight-end
```
</TabItem>

<TabItem value="kaniko">

You can use Kaniko to build container images in Kubernetes. To use Kaniko, you must configure a storage bucket and provide the following in your agent configuration file:

:::note
W&B Launch supports: Amazon S3, Google Cloud Storage (GCS), or Microsoft Azure Blob Storage
:::

1. Set the builder type key (`builder`) to `kaniko`.
2. Provide the URI to a storage bucket for the `build-context-store` key. The launch agent will use the specified storage bucket to store build contexts.
3. **(Optional, but highly recommended)** use identity management tools based on the storage bucket resource you use. For example, we recommend you use [workload identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) if you use GKE and [IAM roles for service accounts](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html) if you use EKS. 

:::note
If you use AKS, you must use [Azure AD Workload Identity](https://learn.microsoft.com/en-us/azure/aks/workload-identity-overview) to provide the agent credentials.
:::

The following agent configuration example uses the Kaniko builder and stores build contexts in an Amazon S3 bucket:

```yaml title="~/.config/wandb/launch-config.yaml"
# W&B entity (i.e. user or team) name
entity: entity-name

# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

# List of queues to poll.
queues:
  - default

// highlight-start
builder:
  type: kaniko
  build-context-store: s3://my-bucket/build-contexts/ 
  build-job-name: wandb-image-build # Kubernetes job name prefix for all builds
// highlight-end
```

Depending on your storage bucket, provide the URI to the `build-context-store` key in one of the following formats:

| Cloud | URI                                                       |
| ----- | --------------------------------------------------------- |
| GCP   | `gs://my-bucket/build-contexts/`                          |
| AWS   | `s3://my-bucket/build-contexts/`                          |
| Azure | `https://my-bucket.blob.core.windows.net/build-contexts/` |


If you run a Kubernetes cluster other than using AKS, EKS, or GKE, you will need to create a Kubernetes secret that contains the credentials for your cloud environment. 

* To grant access to GCP, this secret should contain a [service account json](https://cloud.google.com/iam/docs/keys-create-delete#creating). 
* To grant access to AWS, this secret should contain an [AWS credentials file](https://docs.aws.amazon.com/sdk-for-php/v3/developer-guide/guide_credentials_profiles.html). 


Within your agent configuration file, and within the builder section, set the `secret-name` and `secret-key` keys to let Kaniko use the secrets: 

```yaml
builder:
  type: kaniko
  build-context-store: <my-build-context-store>
  secret-name: <Kubernetes-secret-name>
  secret-key: <secret-file-name>
```

</TabItem>

<TabItem value="noop">

Set the builder type to `noop` (no operation) to prevent your agent from building images.



```yaml title="~/.config/wandb/launch-config.yaml"
# W&B entity (i.e. user or team) name
entity: entity-name

# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

# List of queues to poll.
queues:
  - default

builder:
  type: noop
```

:::tip
Setting the builder type to `noop` is useful if you want to limit the agent to running pre-built container images.
:::

</TabItem>
</Tabs>

### Cloud environments


### Container registries