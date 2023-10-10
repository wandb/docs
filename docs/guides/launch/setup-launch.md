---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up Launch

This page describes the high-level steps required to set up W&B Launch:

1. **Set up a queue**: Queues are FIFO and possess a queue configuration. A queue's configuration controls where and how jobs are are run on a target resource, such as Docker or a Kubernetes cluster.
2. **Set up an agent**: Agents run on your machine/infrastructure and polls one or more queues for jobs. When a job is pulled, the agent ensures that the image is built and available. The agent then submits the job to the target resource.


## Set up a queue
Launch queues must be configured to point to a specific target resource along with any additional configuration specific to that resource.  For example, a Launch queue that is pointing to a kubernetes cluster might specify in its configuration some environment variables or set a custom namespace.  During queue creation, you’ll be prompted to select a resource type and provide a configuration.

When an agent receives a job from a queue, it also receives the queue configuration.  When the agent submits the job to the target resource, it includes the queue configuration along with any overrides from the job itself. For example, job configuration can be used to specify the SageMaker instance type for that job instance only.

### Create a queue
1. Navigate to Launch App at [wandb.ai/launch](https://wandb.ai/launch). 
2. Click the **create queue** button on the top right of the screen. 

![](/images/launch/create-queue.gif)

3. From the **Entity** dropdown menu, select the entity the queue will belong to. 
  :::tip
  If you choose a team entity, all members of the team will be able to send jobs to this queue. Choosing a personal entity (associated with a username) will create a private queue that only that user can use.
  :::
4. Provide a name for your queue in the **Queue** field. 
5. From the **Resource** dropdown, select the compute resource you want jobs added to this queue to use.
6. Provide a resource configuration in either JSON or YAML format in the Configuration field. The structure and semantics of your configuration document will depend on the resource type that the queue is pointing to. For more details, see the dedicated Set up … page[LINK] for your target resource.

<!-- ## Configure the queue
Launch queues are first in, first out (FIFO) queues. When you create a queue, you specify a configuration for that queue. Launch queue use this configuration to figure out where and how execute jobs. The schema of your launch queue configuration depends on the target compute resource you want jobs to be executed on. 

For example, the queue configuration for an Amazon SageMaker queue target resource will differ from that of a Kubernetes cluster queue target resource.

The table below describes the configuration schema for each queue target resource along with links to each target resources' documentation:

| Queue target resource | Configuration schema | Official documentation |
|-----------------------|----------------------|------------------------|
| Docker     | Named arguments to `docker run`. | [Docker CLI documentation](https://docs.docker.com/engine/reference/commandline/run/) |
| SageMaker  | Partial contents of SageMaker API's `CreateTrainingJob` request.| [Amazon SageMaker `CreateTrainingJob` API Reference ](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)
| Kubernetes | Kubernetes Job spec or custom object. | [Kubernetes Job documentation](https://kubernetes.io/docs/concepts/workloads/controllers/job/)


The following tabs show example queue configurations for SageMaker and Kubernetes target resources:

<Tabs
  defaultValue="sagemaker"
  values={[
    {label: 'Sagemaker', value: 'sagemaker'},
    {label: 'Kubernetes', value: 'kubernetes'},
  ]}>
  <TabItem value="sagemaker">
  
We use Amazon SageMaker's `CreateTrainingJob` request schema to define a queue configuration:

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
We use a Kubernetes Job spec schema to define a queue configuration:

```yaml title='Queue configuration in W&B App UI'
kind: Job
spec:
  template:
    spec:
      containers:
        - name: image-name
          image: image-name:latest
      restartPolicy: Never
metadata:
  name: image-name
apiVersion: batch/v1
```

  </TabItem>
</Tabs>


Queue configs can be dynamically configured with macros. These macros are evaluated when the agent dequeues a job from the queue it is polling

launches a job from the queue.  

You can set the following macros:


| Macro             | Description                                           |
|-------------------|-------------------------------------------------------|
| `${project_name}` | The name of the project the run is being launched to. |
| `${entity_name}`  | The owner of the project the run being launched to.   |
| `${run_id}`       | The id of the run being launched.                     |
| `${run_name}`     | The name of the run that is launching.                |
| `${image_uri}`    | The URI of the container image for this run.          |


:::info
Any custom macro, such as `${MY_ENV_VAR}`, is substituted with an environment variable from the agent's environment.
::: -->


## Set up a launch agent
Launch agents are long running processes that poll one or more launch queues for jobs. Launch agents dequeues jobs off of queues in first in, first out (FIFO) order. When an agent dequeues a job from a queue, it potentially builds an image for that job. The agent then submits the job to the target resource along with configuration options provided by the queue configuration.

<!-- Future: Insert image -->

:::info
Agents are highly flexible and can be configured to support a wide variety of use cases.  The required configuration for your agent(s) will depend on your specific use case. See the dedicated [LINK] to learn more.
:::


### Agent configuration
Configure the launch agent with a configuration (config) YAML file named `launch-config.yaml`. By default, W&B will check for the config file in `~/.config/wandb/launch-config.yaml`. You can optionally specify a different directory when you activate the launch agent[LINK].

The contents of your launch agent's configuration file will depend on your launch agent's environment, the launch queue's target resource, Docker builder requirements, cloud registry requirements, and so forth. 

Independent of your use case, there are core configurable options for the launch agent:
* maximum number of jobs the agent can execute in parallel
* W&B entity 
* the name of the queue for the agent to watch.

:::tip
You can use the W&B CLI to specify universal configurable options for the launch agent (instead of the config YAML file): maximum number of jobs, W&B entity, and launch queues. See the `wandb launch-agent` command for more information. [LINK]
:::


The following YAML snippet shows how to specify core launch agent config keys:

```yaml title="launch-config.yaml"
# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

entity: <entity-name>

# List of queues to poll.
queues:
  - default
```

### Configure a container builder
The launch agent can be configured to build images. This is necessary if you intend on using launch jobs created from git repos or code Artifacts (see Creating Launch jobs[LINK]). 

W&B Launch supports two builders:

* Docker: The Docker builder uses a local Docker daemon to build images.
* [Kaniko](https://github.com/GoogleContainerTools/kaniko):  Kaniko is a Google project that enables image building in environments where a Docker daemon is unavailable. 

:::tip
Use the Kaniko builder if your agent is running in an environment where a Docker daemon is unavailable, such as in a Kubernetes cluster. See Launch on Kubernetes for details about the Kaniko builder. [LINK]
:::

To specify an image builder, include the builder key in your agent configuration. For example, the following code snippet shows a portion of the launch config (`launch-config.yaml`) that specifies to use Docker or Kaniko:

```yaml title="launch-config.yaml"
builder:
	type: docker | kaniko
```

### Configure a cloud registry
In some cases, you might want to connect a launch agent to a cloud registry. Common scenarios where this can happen are:

* You want to use the agent to build images and run these images on Amazon SageMaker or VertexAI.
* You want the launch agent to provide credentials to pull from an image repository.

To learn more about this see [LINK].

## Activate the launch agent
Activate the launch agent with the `launch-agent`[LINK] W&B CLI command:

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

In some use cases, you might want to have a launch agent polling queues from within a Kubernetes cluster. See the [LINK] for more information. 
















<!-- OLD -->
<!-- Start -->
<!-- W&B Launch uses a launch agent to poll one or more launch queues for launch jobs. The launch agent will remove launch jobs from the queue (FIFO) and execute them based on the queue's target resource (defined in the W&B App UI) and the agent configuration file you define in a YAML file called `launch-config.yaml`. By default, the agent configuration file is stored in `~/.config/wandb/launch-config.yaml`. 

When the launch agent removes a launch job from the queue, it will build a container image for that launch job. By default, W&B will build the image with Docker. See the [Container builder section](#container-builder-options) for alternative Docker image builders.

The environment that a launch agent is running in, and polling for launch jobs, is called the *agent environment*. Example agent environments include: locally on your machine or Kubernetes clusters. See the [Launch agent environments](#launch-agent-environments) section for more information.

Depending on the compute target resource of the queue and your Docker builder, you might need to specify a container registry for the launch agent to store container images. See the [Container registries](#container-registries) section for more information.




:::tip
The contents of your agent configuration file (`~/.config/wandb/launch-config.yaml`) depend on the launch queue's target resource. At a minimum, you must specify your W&B entity, the maximum number of jobs, and the name of the queue to use for the `entity`, `max_jobs`, and `queues` keys, respectively.
:::

:::note
The launch agent environment is independent of a queue's launch target resource.
:::

The following YAML snippet lists the required keys you must specify in your agent configuration:

```yaml title="~/.config/wandb/launch-config.yaml"
# W&B entity (i.e. user or team) name
entity: entity-name

# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

# List of queues to poll.
queues:
  - default
```


### Container builder options
Launch agents poll queues and send launch jobs (as a Docker image) to the compute resource that was configured for the launch queue. You can optionally specify the container builder (`builder` key) you want to use in your launch agent configuration.

:::info
The `builder` key is not required and, if omitted, the agent will use Docker to build images locally.
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

You can use Kaniko to build container images in Kubernetes. To use Kaniko, you must configure:

* a storage bucket
* specify a registry key in the agent config 
* specify an environment key in the agent config  


And provide the following in your agent configuration file:

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

// highlight-start
builder:
  type: noop
// highlight-end
```

:::tip
Setting the builder type to `noop` is useful if you want to limit the agent to running pre-built container images.
:::

</TabItem>
</Tabs>


### Launch agent environments
Within your launch agent configuration, you can specify the agent environment. The agent environment is the environment where a launch agent is running, polling for jobs. The following table describes the possible agent environments you can use based on your target resource:

| Target resource |    Possible Launch agent environments     |
| --------------- | ----------------------------------------- |
| SageMaker       |   Local machine, AWS, GCP, Azure  |
| Docker          |   Local machine, AWS, GCP, Azure  |
| Kubernetes      |   Kubernetes cluster              |


Based on your use case, see the following tabs for information on how to specify the agent environment in your launch agent configuration file (`~/.config/wandb/launch-config.yaml`).

<Tabs
defaultValue="aws"
values={[
{label: 'AWS', value: 'aws'},
{label: 'GCP', value: 'gcp'},
{label: 'Azure', value: 'azure'}
]}>

<TabItem value="aws">

The AWS environment configuration requires the `region` key to be set. The region should be the AWS region that the agent will be running in. When the agent starts, it will use `boto3` to load the default AWS credentials. See the [boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#overview) for more information on how to configure default AWS credentials.

```yaml
environment:
  type: aws
  region: <aws-region>
```

</TabItem>

<TabItem value="gcp">

The GCP environment requires the `region` and `project` keys to be set. The region should be the GCP region that the agent will be running in. The project should be the GCP project that the agent will be running in. When the agent starts, it will use `google.auth.default()` to load the default GCP credentials. See the [google-auth documentation](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) for more information on how to configure default GCP credentials.

```yaml
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
```

</TabItem>

<TabItem value="azure">

The Azure environment does not require any additional keys to be set. When the agent starts, it will use `azure.identity.DefaultAzureCredential()` to load the default Azure credentials. See the [azure-identity documentation](https://docs.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) for more information on how to configure default Azure credentials.

```yaml
environment:
  type: azure
```

</TabItem>

</Tabs>

### Container registries
The `registry` key is used to configure the container registry that the agent will use to store container images. If the agent does not require access to a container registry, this key should be omitted.

<Tabs
defaultValue="ecr"
values={[
{label: 'AWS ECR', value: 'ecr'},
{label: 'GCP Artifact Registry', value: 'gcr'},
{label: 'Azure Container Registry', value: 'acr'}
]}>

<TabItem value="ecr">

Set `type: ecr` to use AWS Elastic Container Registry. In order to use `ecr`, must configure an AWS environment. The `repository` key is required and should be the name of the ECR repository that the agent will use to store container images. The agent will use the region configured in the `environment` key to determine which registry to use.

```yaml
registry:
  # Requires an aws environment configuration.
  type: ecr
  # URI of the ECR repository where the agent will store images.
  # Make sure the region matches what you have configured in your
  # environment.
  uri: <account-id>.ecr.<region>.amazonaws.com/<repository-name>
  # Alternatively, you can simply set the repository name
  # repository: my-repository-name
```

</TabItem>
<TabItem value="gcr">

Set `type: gcr` to use GCP Artifact Registry. In order to use `gcr`, must configure a GCP environment. The `repository` and `image-name` keys are required. The `repository` key should be the name of the Artifact Registry repository that the agent will use to store container images. The agent will use the region and project configured in the `environment` key to determine which registry to use. The `image-name` key should be the name of the image within the repository that the agent will use to store container images.

```yaml
registry:
  # Requires a gcp environment configuration.
  type: gcr
  # URI of the Artifact Registry repository and image name where the agent
  # will store images. Make sure the region and project match what you have
  # configured in your environment.
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Alternatively, you may set the repository and image-name keys.
  # repository: my-artifact-repo
  # image-name: my-image-name
```

</TabItem>
<TabItem value="acr">

Set `type: acr` to use Azure Container Registry. In order to use `acr`, must configure an Azure environment. You must set the `uri` key to the URI of a repository within an Azure Container Registry. The agent will push images to the repository specified in the URI.

```yaml
registry:
  type: acr
  uri: https://my-registry.azurecr.io/my-repository
```

</TabItem>
</Tabs> -->






<!-- 
## Permissions 
The launch agent requires access to a cloud environment if it is configured to use a cloud environment. The agent will search for specific credentials based on the `environment` key defined in the `~/.config/wandb/launch-config.yaml`. The agent will use these credentials to push and pull container images, access cloud storage, and trigger on demand compute through cloud services.

For example, if you specified `environment: aws`, the launch agent will look for for an IAM Role[LINK] that grants access to an Amazon S3 bucket.

<Tabs
defaultValue="aws"
values={[
{label: 'AWS IAM role', value: 'aws'},
{label: 'GCP', value: 'gcp'},
{label: 'Azure', value: 'azure'}
]}>

<TabItem value="aws">

<!-- 
The agent requires credentials to access ECR to push and pull container images if you are using a builder, and access to an S3 bucket if you want to use the kaniko builder. The following policy can be used to grant the agent access to ECR and S3.  -->
<!-- 
Ensure your launch agent has access to the Amazon Elastic Container Registry (ECR). To do so, create an IAM role[LINK] and attach the following permissions (see code snippet below). Replace content within `<>` with your appropriate values:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:PutLifecyclePolicy",
        "ecr:PutImageTagMutability",
        "ecr:StartImageScan",
        "ecr:CreateRepository",
        "ecr:PutImageScanningConfiguration",
        "ecr:UploadLayerPart",
        "ecr:BatchDeleteImage",
        "ecr:DeleteLifecyclePolicy",
        "ecr:DeleteRepository",
        "ecr:PutImage",
        "ecr:CompleteLayerUpload",
        "ecr:StartLifecyclePolicyPreview",
        "ecr:InitiateLayerUpload",
        "ecr:DeleteRepositoryPolicy"
      ],
      "Resource": "arn:aws:ecr:<REGION>:<ACCOUNT-ID>:repository/<YOUR-REPO-NAME>"
    },
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "ecr:BatchCheckLayerAvailability",
      "Resource": "arn:aws:ecr:<REGION>:<ACCOUNT-ID>:repository/<YOUR-REPO-NAME>"
    },
    {
      "Sid": "ListObjectsInBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>"]
    },
    {
      "Sid": "AllObjectActions",
      "Effect": "Allow",
      "Action": "s3:*Object",
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>/*"]
    }
  ]
}
```

In order to use SageMaker queues, you will also need to create a separate execution role that is assumed by your jobs running in Amazon SageMaker. The agent should be granted the following permissions to be allowed to create training jobs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sagemaker:CreateTrainingJob",
      "Resource": "arn:aws:sagemaker:<REGION>:<ACCOUNT-ID>/*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "<ARN-OF-ROLE-TO-PASS>"
    },
    {
      "Effect": "Allow",
      "Action": "kms:CreateGrant",
      "Resource": "<ARN-OF-KMS-KEY>",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "sagemaker.<REGION>.amazonaws.com",
          "kms:GrantIsForAWSResource": "true"
        }
      }
    }
  ]
}
```

:::note
The `kms:CreateGrant` permission for SageMaker queues is required only if the associated ResourceConfig has a specified VolumeKmsKeyId and the associated role does not have a policy that permits this action.
:::

</TabItem>

<TabItem value="gcp">

The gcp credentials in the agents environment must include the following permissions in order to access various services:

```yaml
# Permissions for accessing cloud storage. Required for kaniko build.
storage.buckets.get
storage.objects.create
storage.objects.delete
storage.objects.get

# Permissions for accessing artifact registry. Required for any container build.
artifactregistry.dockerimages.list
artifactregistry.repositories.downloadArtifacts
artifactregistry.repositories.list
artifactregistry.repositories.uploadArtifacts

# Permissions for listing accessible compute regions. Required always.
compute.regions.get

# Permissions for managing Vertex AI jobs. Required for Vertex queues.
ml.jobs.create
ml.jobs.list
ml.jobs.get
```

</TabItem>

<TabItem value="azure">

The agent requires the following roles be assigned to its service principal in order to access various services:

- **Storage Blob Data Contributor**: Required for kaniko build ([docs](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor))
- **AcrPush**: Required for kaniko build ([docs](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-roles#acrpush))

These permissions should be scoped to any storage containers and containers registries that you plan to access with the agent.

</TabItem>

</Tabs> --> -->
