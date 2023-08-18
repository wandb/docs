---
description: Getting started guide for W&B Launch.
displayed_sidebar: default
---

# Walkthrough

Follow this guide to get started using W&B Launch. This guide will walk you through the setup of the fundamental components of a launch workflow: a **job**, **launch queue**, and **launch agent**. Specifically, you will create a job that trains a neural network, create a launch queue that will be used to submit jobs for execution on your local machine, and then create a launch agent that will poll on the queue and execute the jobs that it pops from the queue with Docker.

:::info
Ensure you have W&B Python SDK version 0.14.0 or higher by running ```
wandb --version```.
:::

## Before you get started
Before you get started, install Docker on the machine where you will run your launch agent. See the [Docker documentation](https://docs.docker.com/get-docker/) for more information on how to install Docker, and make sure the docker daemon is running on your machine before you proceed.

## Create a job

Jobs are created automatically from any W&B run that has associated source code. For more details on how source code can be associated with a run, see [these docs](create-job.md).

Copy the following Python code to your machine in a file named `train.py`.

```python
import wandb

config = {
    "epochs": 10
}

with wandb.init(config=config, project="launch-quickstart"):
    config = wandb.config
    for epoch in range(1, config.epochs):
        loss = config.epochs / epoch
        accuracy = (1 + (epoch / config.epochs))/2
        wandb.log({
            "loss": loss, 
            "accuracy": accuracy, 
            "epoch": epoch})
    wandb.run.log_code()  

```

:::note
By default, `Run.log_code()` ignores all paths under the `wandb` library's metadata directories. (`<project_root>/.wandb` and `<project_root/wandb`)
:::

To install dependencies and run the script, execute the following commands in your terminal:

```bash
pip install wandb>=0.14.0
python train.py
```

Let the script run to completion and then move on to the next step. Your console output should look roughly like:

```
wandb: Syncing run trim-planet-100
wandb: ⭐️ View project at https://wandb.ai/bcanfieldsherman/uncategorized
wandb: 🚀 View run at https://wandb.ai/bcanfieldsherman/uncategorized/runs/5av9db29
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb: accuracy ▁▂▃▄▅▅▆▇█
wandb:    epoch ▁▂▃▄▅▅▆▇█
wandb:     loss █▄▃▂▂▁▁▁▁
wandb: 
wandb: Run summary:
wandb: accuracy 0.95
wandb:    epoch 9
wandb:     loss 1.11111
wandb: 
wandb: 🚀 View run trim-planet-100 at: https://wandb.ai/bcanfieldsherman/uncategorized/runs/5av9db29
wandb: Synced 4 W&B file(s), 0 media file(s), 9 artifact file(s) and 1 other file(s)
```

Navigate to your new **launch-quickstart** project in your W&B account and open the jobs tab from the nav on the left side of the screen.

![](/images/launch/jobs-tab.png)

The **Jobs** page displays a list of W&B Jobs that were created from previously executed W&B Runs. You should see a job named **job-source-launch-quickstart-train.py:v0**. You can edit the name of the job from the jobs page if you would like to make the job a bit more memorable. Click on your job to open a detailed view of your job including the source code and dependencies for the job and a list of runs that have been launched from this job.

## Queue your job

Head back to the page for your job. It should look something like the image below:

![](/images/launch/simple-job.png)

Click the **Launch** button in the top right to launch a new run from this job. A drawer will slide from the right side of your screen and present you with some options for your new run:

* **Job version**: the version of the job to launch. Jobs are versioned like any other W&B Artifact. Different versions of the same job will be created if you make modifications to the software dependencies or source code used to run the job. Since we only have one version, we will select the default **@latest** version.
* **Overrides**: new values for any of jobs inputs. These can be used to change the entrypoint command, arguments, or values in the `wandb.config` of your new run. Our run had one value in the `wandb.config`: `epochs`. We can override this value by in the overrides field. We can also paste values from other runs using this job by clicking the **Paste from...** button.
* **Queue**: the queue to launch the run on. If you have not created any queues yet, you should have the option to create a **Starter Queue**. This queue will be used to launch runs on your local machine using Docker.

![](/images/launch/starter-launch.gif)

Once you have configured your job as desired, click the **launch now** button at the bottom of the drawer to enqueue your launch job.

## Start a launch agent

To execute your job, you will need to start a launch agent polling on your launch queue.

1. From [wandb.ai/launch](https://wandb.ai/launch) navigate to the page for your launch queue.
2. Click the **Add an agent** button.
3. A modal will appear with a W&B CLI command. Copy this and paste this command into your terminal.

![](/images/launch/activate_starter_queue_agent.png)

In general, the command to start a launch agent is:

```bash
wandb launch-agent -e <entity-name> -q <queue-name>
```

Within your terminal, you will see the agent begin to poll for queues. The agent should pick up the job you enqueued earlier and begin to execute. First, the agent will build a container image from the job version you selected. Then, the agent will execute the job on its local host via `docker run`.

That’s it! Navigate to your Launch workspace or your terminal to see the status of your launch job. Jobs are executed in first in, first out order (FIFO). All jobs pushed to the queue will use the same resource type and resource arguments.
