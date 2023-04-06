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


## Agent configuration

The launch agent can be configured with a variety of flags and options. The agent can be configured with a config file or with command line flags. The config file is located at `~/.config/wandb/launch-config.yaml` by default, but the location of the config can be overridden with the `--config` flag. The config file is a YAML file with the following structure:

```yaml
base_url: https://api.wandb.ai # URL of the W&B API server
entity: <entity-name>  # W&B entity (i.e. user or team) name
max_jobs: -1 # Set max concurrent jobs here, -1 = no limit
queues:
- default # Set queues to poll by name here
environment:
  type: aws|gcp
registry:
  type: ecr|gcr
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

```yaml
environment:
  type: aws
  region: us-east-2
```

</TabItem>

<TabItem value="gcp">

```yaml
environment:
  type: gcp
  region: us-east1
  project: my-gcp-project
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

```yaml
registry:
  type: ecr  # requires an aws environment configuration
  repository: my-ecr-repo.  # name of repository in the ecr for the region configured in your environment
```

</TabItem>

<TabItem value="gcr">

```yaml
registry:
  type: gcr  # requires a gcp environment configuration
  repository: my-artifact-repo  # name of artifact repository in project/region configured in your environment
  image-name: my-image-name # the name (not tag!) of image within repository
```

</TabItem>

</Tabs>


### Builders

The `builder` key is used to configure the container builder that the agent will use to build container images. If the agent does not require access to a container builder, this key should be omitted.

<Tabs
  defaultValue="docker"
  values={[
    {label: 'Docker', value: 'docker'},
    {label: 'Kaniko', value: 'kaniko'},
    {label: 'Noop', value: 'noop'}
  ]}>

<TabItem value="docker">

```yaml
builder:
  type: docker
```

</TabItem>

<TabItem value="kaniko">

```yaml
builder:
  type: kaniko
  build-context-store: s3://my-bucket/build-contexts/  # s3 or gcs prefix for build context storage
  build-job-name: wandb-image-build  # k8s job name prefix for all builds
```
</TabItem>

<TabItem value="noop">

The `noop` builder is useful if you want to limit the agent to running pre-built container images.

```yaml
builder:
  type: noop
```

</TabItem>

</Tabs>