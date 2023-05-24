---
description: Launch agent documentation
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Start an agent

Start a launch agent to execute jobs from your queues.

## Overview

A **launch agent** is a long-running process that polls on one or more launch queues and executes the jobs that it pops from the queue. A launch agent can be started with the `wandb launch-agent` command and is capable on launching jobs onto a multitude of compute platforms, including docker, kubernetes, sagemaker, and more. The launch agent is implemented entirely in our [open source client library](https://github.com/wandb/wandb/tree/main/wandb/sdk/launch).

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


## Agent configuration

The launch agent can be configured with a variety of flags and options. The agent can be configured with a config file or with command line flags. The config file is located at `~/.config/wandb/launch-config.yaml` by default, but the location of the config can be overridden with the `--config` flag. The config file is a YAML file with the following structure:

```yaml
# URL of your W&B server. Defaults to https://api.wandb.ai
base_url: https://api.wandb.ai

# W&B entity (i.e. user or team) name
entity: <entity-name>

# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

# List of queues to poll.
queues:
- default

# Cloud environment config.
environment:
  type: aws|gcp

# Container registry config.
registry:
  type: ecr|gcr

# Container build config.
builder:
  type: docker|kaniko|noop
```

The `environment`, `registry`, and `builder` keys are optional. By default, the agent will use no cloud environment, a local docker registry, and a docker builder. 

### Environments

The `environment` key is used to configure the cloud environment that the agent will need to access in order to do its job. If the agent does not require access to cloud resources, this key should be omitted. See the following references for configuring an AWS or GCP environment.

<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'GCP', value: 'gcp'},
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

</Tabs>

### Registries

The `registry` key is used to configure the container registry that the agent will use to store container images. If the agent does not require access to a container registry, this key should be omitted.

<Tabs
  defaultValue="ecr"
  values={[
    {label: 'AWS ECR', value: 'ecr'},
    {label: 'GCP Artifact Registry', value: 'gcr'},
  ]}>

<TabItem value="ecr">

Set `type: ecr` to use AWS Elastic Container Registry. In order to use `ecr`, must configure an AWS environment. The `repository` key is required and should be the name of the ECR repository that the agent will use to store container images. The agent will use the region configured in the `environment` key to determine which registry to use.

```yaml
registry:
  # Requires an aws environment configuration.
  type: ecr
  # Name of ECR repository in region configured 
  # in your environment.
  repository: my-ecr-repo.
```

</TabItem>

<TabItem value="gcr">

Set `type: gcr` to use GCP Artifact Registry. In order to use `gcr`, must configure a GCP environment. The `repository` and `image-name` keys are required. The `repository` key should be the name of the Artifact Registry repository that the agent will use to store container images. The agent will use the region and project configured in the `environment` key to determine which registry to use. The `image-name` key should be the name of the image within the repository that the agent will use to store container images.

```yaml
registry:
  # Requires a gcp environment configuration.
  type: gcr
  # Name of artifact repository in project/region configured in your environment.
  repository: my-artifact-repo
  # Name (not tag!) of image within repository where the agent will store images.
  image-name: my-image-name
```

</TabItem>

</Tabs>


### Builders

The `builder` key is used to configure the container builder that the agent will use to build container images. The `builder` key is not required and if omitted, the agent will use Docker to build images locally.

<Tabs
  defaultValue="docker"
  values={[
    {label: 'Docker', value: 'docker'},
    {label: 'Kaniko', value: 'kaniko'},
    {label: 'Noop', value: 'noop'}
  ]}>

<TabItem value="docker">

Set `type: docker` to use Docker to build images locally. The `docker` builder is selected by default and does not require any additional configuration.

```yaml
builder:
  type: docker
```

</TabItem>

<TabItem value="kaniko">

Set `type: kaniko` and configure either a GCP or AWS environment to use Kaniko to build container images in Kubernetes. The `kaniko` builder requires the `build-context-store` key to be set. The `build-context-store` key should be an S3 or GCS prefix that the agent will use to store build contexts, depending on the environment that is configured. The `build-job-name` can be used to specify a name prefix for the k8s job that the agent will use to build images.

```yaml
builder:
  type: kaniko
  build-context-store: s3://my-bucket/build-contexts/  # s3 or gcs prefix for build context storage
  build-job-name: wandb-image-build  # k8s job name prefix for all builds
```

To grant Kaniko builds running in k8s access to your blob and container storage in the cloud, it is strongly recommended that you use either [workload identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) if you are using GKE or [IAM roles for service accounts](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html) if you are using EKS.

If you are running your own k8s cluster, you will need to create a k8s secret that contains the credentials for your cloud environemnt. To grant access to GCP, this secret should contain a [service account json](https://cloud.google.com/iam/docs/keys-create-delete#creating). To grant access to AWS, this secret should contain an [AWS credentials file](https://docs.aws.amazon.com/sdk-for-php/v3/developer-guide/guide_credentials_profiles.html). You can configure the Kaniko builder to use this secret by setting the following keys in your builder config:

```yaml
builder:
  type: kaniko
  build-context-store: <my-build-context-store>
  secret-name: <k8s-secret-name>
  secret-key: <secret-file-name>
```

</TabItem>

<TabItem value="noop">

The `noop` builder is useful if you want to limit the agent to running pre-built container images. Simply set `type: noop` to prevent your agent from building images.

```yaml
builder:
  type: noop
```

</TabItem>

</Tabs>

## View agents

Navigate to the page for a particular launch queue, and the navigate to the **Agents** tab to view active and inactive agents assigned to the queue. Within this tab you can view the:

- **Agent ID**:  Unique agent identification
- **Status:** The statue of the agent. An agent can have a **Killed** or **Polling** status.
- **Start date:** The date the agent was activated.
- **Host:** The machine the agent is polling on.

![](/images/launch/queues_all_agents.png)
