---
description: Getting started guide for W&B Launch.
menu:
  launch:
    identifier: walkthrough
    parent: launch
title: 'Tutorial: W&B Launch basics'
url: guides/launch/walkthrough
weight: 1
---
## What is Launch? 

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Easily scale training [runs]({{< relref "/guides/models/track/runs/" >}}) from your desktop to a compute resource like Amazon SageMaker, Kubernetes and more with W&B Launch. Once W&B Launch is configured, you can quickly run training scripts, model evaluation suites, prepare models for production inference, and more with a few clicks and commands. 

## How it works

Launch is composed of three fundamental components: **launch jobs**, **queues**, and **agents**.

A [*launch job*]({{< relref "./launch-terminology.md#launch-job" >}}) is a blueprint for configuring and running tasks in your ML workflow. Once you have a launch job, you can add it to a [*launch queue*]({{< relref "./launch-terminology.md#launch-queue" >}}). A launch queue is a first-in, first-out (FIFO) queue where you can configure and submit your jobs to a particular compute target resource, such as Amazon SageMaker or a Kubernetes cluster. 

As jobs are added to the queue, [*launch agents*]({{< relref "./launch-terminology.md#launch-agent" >}}) poll that queue and execute the job on the system targeted by the queue.

{{< img src="/images/launch/launch_overview.png" alt="W&B Launch overview diagram" >}}

Based on your use case, you (or someone on your team) will configure the launch queue according to your chosen [compute resource target]({{< relref "./launch-terminology.md#target-resources" >}}) (for example Amazon SageMaker) and deploy a launch agent on your own infrastructure. 

See the [Terms and concepts]({{< relref "./launch-terminology.md" >}}) page for more information about Launch.

## How to get started

Depending on your use case, explore the following resources to get started with W&B Launch:

* If this is your first time using W&B Launch, we recommend you go through the [Launch walkthrough]({{< relref "#walkthrough" >}}) guide.
* Learn how to set up [W&B Launch]({{< relref "/launch/set-up-launch/" >}}).
* Create a [launch job]({{< relref "./create-and-deploy-jobs/create-launch-job.md" >}}).
* Check out the W&B Launch [public jobs GitHub repository](https://github.com/wandb/launch-jobs) for templates of common tasks like [deploying to Triton](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton), [evaluating an LLM](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals), or more.
    * View launch jobs created from this repository in this public [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B project.

## Walkthrough

This page walks through the basics of the W&B Launch workflow.

{{% alert %}}
W&B Launch runs machine learning workloads in containers. Familiarity with containers is not required but may be helpful for this walkthrough. See the [Docker documentation](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/) for a primer on containers.
{{% /alert %}}

## Prerequisites

Before you get started, ensure you have satisfied the following prerequisites:

1. Sign up for an account at https://wandb.ai/site and then log in to your W&B account. 
2. This walkthrough requires terminal access to a machine with a working Docker CLI and engine. See the [Docker installation guide](https://docs.docker.com/engine/install/) for more information. 
3. Install W&B Python SDK version `0.17.1` or higher:
```bash
pip install wandb>=0.17.1
```
4. Within your terminal, execute `wandb login` or set the `WANDB_API_KEY` environment variable to authenticate with W&B.

{{< tabpane text=true >}}
{{% tab "Log in to W&B" %}}
    Within your terminal execute:
    
    ```bash
    wandb login
    ```
{{% /tab %}}
{{% tab "Environment variable" %}}

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    Replace `<your-api-key>` with your W&B API key.
{{% /tab %}}
{{% /tabpane %}}

## Create a launch job
Create a [launch job]({{< relref "./launch-terminology.md#launch-job" >}}) in one of three ways: with a Docker image, from a git repository or from local source code:

{{< tabpane text=true >}}
{{% tab "With a Docker image" %}}
To run a pre-made container that logs a message to W&B, open a terminal and run the following command:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

The preceding command downloads and runs the container image `wandb/job_hello_world:main`. 

Launch configures the container to report everything logged with `wandb` to the `launch-quickstart` project. The container logs a message to W&B and displays a link to the newly created run in W&B. Click the link to view the run in the W&B UI.
{{% /tab %}}
{{% tab "From a git repository" %}}
To launch the same hello-world job from its [source code in the W&B Launch jobs repository](https://github.com/wandb/launch-jobs), run the following command:

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
The command does the following:
1. Clone the [W&B Launch jobs repository](https://github.com/wandb/launch-jobs) to a temporary directory.
2. Create a job named **hello-world-git** in the **hello** project. This job tracks the exact source code and configuration used to run execute the code.
3. Build a container image from the `jobs/hello_world` directory and the `Dockerfile.wandb`.
4. Start the container and run the `job.py` python script.

The console output shows the image build and execution. The output of the container should be nearly identical to the previous example.

{{% /tab %}}
{{% tab "From local source code" %}}

Code not versioned in a git repository can be launched by specifying a local directory path to the `--uri` argument. 

Create an empty directory and add a Python script named `train.py` with the following content:

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

Add a file `requirements.txt` with the following content:

```text
wandb>=0.17.1
```

From within the directory, run the following command:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

The command does the following:
1. Log the contents of the current directory to W&B as a Code Artifact.
2. Create a job named **hello-world-code** in the **launch-quickstart** project.
3. Build a container image by copying `train.py` and `requirements.txt` into a base image and `pip install` the requirements.
4. Start the container and run `python train.py`.
{{% /tab %}}
{{< /tabpane >}}

## Create a queue

Launch is designed to help teams build workflows around shared compute. In the examples so far, the `wandb launch` command has executed a container synchronously on the local machine. Launch queues and agents enable asynchronous execution of jobs on shared resources and advanced features like prioritization and hyperparameter optimization. To create a basic queue, follow these steps:

1. Navigate to [wandb.ai/launch](https://wandb.ai/launch) and click the **Create a queue** button.
2. Select an **Entity** to associate the queue with. 
3. Enter a **Queue name**.
4. Select **Docker** as the **Resource**.
5. Leave **Configuration** blank, for now.
6. Click **Create queue** :rocket:

After clicking the button, the browser will redirect to the **Agents** tab of the queue view. The queue remains in the **Not active** state until an agent starts polling.

{{< img src="/images/launch/create_docker_queue.gif" alt="Docker queue creation" >}}

For advanced queue configuration options, see the [advanced queue setup page]({{< relref "/launch/set-up-launch/setup-queue-advanced.md" >}}).

## Connect an agent to the queue

The queue view displays an **Add an agent** button in a red banner at the top of the screen if the queue has no polling agents. Click the button to view copy the command to run an agent. The command should look like the following:

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

Run the command in a terminal to start the agent. The agent polls the specified queue for jobs to run. Once received, the agent downloads or builds and then executes a container image for the job, as if the `wandb launch` command was run locally.

Navigate back to [the Launch page](https://wandb.ai/launch) and verify that the queue now shows as **Active**.

## Submit a job to the queue

Navigate to your new **launch-quickstart** project in your W&B account and open the jobs tab from the navigation on the left side of the screen.

The **Jobs** page displays a list of W&B Jobs that were created from previously executed runs. Click on your launch job to view source code, dependencies, and any runs created from the job. After completing this walkthrough there should be three jobs in the list.


Pick one of the new jobs and follow these instructions to submit it to the queue:

1. Click the **Launch** button to submit the job to a queue. The **Launch** drawer will appear. 
2. Select the **Queue** you created earlier and click **Launch**. 

This submits the job to the queue. The agent polling this queue picks up and executes the job. The progress of the job can be monitored from the W&B UI or by inspecting the output of the agent in the terminal.

The `wandb launch` command can push jobs to the queue directly by specifying the `--queue` argument. For example, to submit the hello-world container job to the queue, run the following command:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```