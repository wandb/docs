---
description: Discover how to create a job for W&B Launch.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a job

A job is a complete blueprint of how to perform a step in your ML workflow, like training a model, running an evaluation, or deploying a model to an inference server. Each job contains contextual information about the run it is being created from, including the source code, entrypoint, software dependencies, hyperparameters, dataset version, and so on.


:::info
`wandb>=0.13.8` is required in order to create jobs.
:::

Once you have one or more jobs created, you can add them to a [launch queue](./create-queue.md). W&B will then execute your jobs based on your queue configuration. See the [Launch a job](./launch-jobs.md) section for information on how to add jobs to your queue.


:::tip
Similar to [launch queues](./create-job.md), a launch job belongs to a W&B entity. Both the launch queue and the jobs you want to enqueue must belong to the same W&B entity.
:::




## How do I create a job?

Jobs are captured automatically from runs that you track with W&B if you also track the run's source code. When you launch a job created this way, W&B will automatically build and run a Docker image using that code, the associated requirements.txt, and the base image associated with the queue on which the job is launched. 

Alternatively, you can create a job using a Docker image if you prefer to bake in dependencies and custom base images.

:::info
* Conda environments are not yet supported for capturing job dependencies. Use the Docker image job creation path if you manage dependencies with Conda today. 
* See [this section](docker#docker-queues) for more information on choosing an accelerator base image.
* Launch's automated container build requires that the agent runs with access to the Docker CLI or in an EKS or GKE cluster with [kaniko configured](run-agent#builders). If you plan to run your agent in a way that is incompatible with these build requirements, use the Docker image job creation path and set `builder.type` to `noop` in your launch config.
* Launch does not support custom Dockerfiles. Instead, specify a Docker image that contains the resources necessary to create and execute a W&B job. W&B will use the Docker image you specify and create an image sourced job. Use a docker image sourced job from a repo that your cluster has access to. For more info about making jobs from images, see the ["Docker image" tab in the Create a job page](./create-job.md#how-do-i-create-a-job).
:::


<Tabs
  defaultValue="git"
  values={[
    {label: 'Git', value: 'git'},
    {label: 'Code artifact', value: 'artifact'},
    {label: 'Docker image', value: 'docker'},
  ]}>

<TabItem value="artifact">

Log code to your run as an Artifact by calling `run.log_code()`.
Example:

```python
run = wandb.init()
run.log_code(".", include_fn=lambda path: path.endswith(".py"))
```

</TabItem>

<TabItem value="git">

Associate your run with a git commit from your git repo. W&B will automatically capture a commit id and diff when `wandb.init()` is called by code in a git repository cloned from a remote url. If your repository contains a `requirements.txt` file, the launch agent will install any specified dependencies when running your job.

</TabItem>

<TabItem value="docker">

Associate your run with a Docker image. W&B will look for an image tag in the `WANDB_DOCKER` environment variable, and if `WANDB_DOCKER` is set, a job will be created from the specified image tag. When you launch a job from a Docker image, the launch agent run the image specified.

For more information on `WANDB_DOCKER` see the docs for our [docker integration](../integrations/other/docker.md).

Be sure to set the `WANDB_DOCKER` environment variable to the full image tag as you it will be accessible to your launch agent. For example, if your agent will run images from an ECR repository, you should set `WANDB_DOCKER` to the full image tag, including the ECR repository URL, e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:develop`. The docker tag, in this case 'develop', will be added as an alias to the resulting job. Creating more launch jobs from this image results in newer versions of the job, with updated aliases. 

To create your first container image-sourced job, simply run:
```bash
docker run -e WANDB_DOCKER=<image-tag> <image-tag>
```

</TabItem>

</Tabs>

## Job naming conventions

By default, W&B automatically creates a job name for you. The name is generated depending on how the job is created (GitHub, code artifact, or Docker image). The following table describes the job naming convention used for each job source:

| Source        | Naming convention                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |


:::note
For docker image jobs, the image tag is automatically added as an alias to the job.
:::
## Make your code job-friendly

Jobs are parameterized by the values in your `wandb.config`. When your job is executed and `wandb.init` is called, your `wandb.config` will be populated with the values specified for this run. This means it's important that you use the `wandb.config` to store and access all of the parameters that you want to be able to vary between runs.

If you read and store your parameters using a different method, you can still use W&B Launch to run your job, you will need to update your code to use the `wandb.config`.

For example, if you use `argparse` to parse your command line arguments, you can easily adapt your script to use the `wandb.config` to store the values of your arguments. For example:

```python
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# Pass the arguments to wandb.init to store them in 
# wandb.config and allow the launch agent to modify them. 
wandb.init(config=args)

# Use the values in wandb.config instead of args. 
# The wandb.config supports dot and dictionary 
# style access, e.g. wandb.config.learning_rate or 
# wandb.config["learning_rate"].
args = wandb.config
```

## View your jobs
View details of launched jobs with the W&B App.

### View details of each job

Navigate to your W&B Project to view fine-grained details of each job such as runs created by a job, the full name of your jobs, and version metadata associated with a project. 

1. Navigate to your W&B project.
2. Select the **Jobs** icon on the left sidebar.
3. A **Jobs** page will appear. In it, you can view all of the jobs created in that project.

![](/images/launch/view_jobs.png)

For example, in the following image we have two job listed:
- **job-https___github.com_githubrepo_demo_launch.git_canonical_job_example.py**
- **job-source-launch_demo-canonical_job_example.py**

Select a job from the list to learn more about that job. A new page with a list of runs created by the job, along with job and version details will appear.  This information is contained in three tabs: **Runs**, **Job details**, and **Version details**.

<Tabs
  defaultValue="runs"
  values={[
    {label: 'Runs', value: 'runs'},
    {label: 'Job details', value: 'jobs_details'},
    {label: 'Version details', value: 'version_details'},
  ]}>
  <TabItem value="runs">

The Runs tab provides information about each run created by the job such as the:

- **Run**: The name of the run.
- **State**: The state of the run.
- **Job version**: The version of the job used.
- **Creator**: Who created the run.
- **Creation date**: The timestamp of when the run was started.
- **Other**: The remaining columns will contain the key-value pairs of your `wandb.config`.

![](/images/launch/runs_in_job.png)


  </TabItem>
  <TabItem value="jobs_details">

The **Job details** provides information about:

* **Description**: An optional description of the job. Select the pencil icon next to this field to add a description.
* **Owner entity**: The entity the job belongs to.
* **Parent project**: The project the job belongs to.
* **Full name**: The full name of your job
* **Creation date**: Creation date of the job.


![](/images/launch/job_id_full_name.png)

  </TabItem>
  <TabItem value="version_details">

Use the **Version details** tab to view specific information about each job version such as the input and output types, and files used for each job version. 

![](/images/launch/version_details_large.png)

  </TabItem>
</Tabs>
