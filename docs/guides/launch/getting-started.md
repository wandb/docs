---
description: Getting started guide for W&B Launch.
---
# Getting started

Follow this getting started guide to launch your first job. This guide will get you up and running using a demo “starter” queue that uses a Docker container as its compute resource.

:::tip
In most enterprise settings, it is expected that the MLOps Team will create queues and configure those queues to infrastructure and external cloud resources. In addition, it is expected that the MLOps team will activate and manage launch agents.  The ML Practitioner can then add jobs to those queues.
:::

There are three major steps outlined in this getting started guide:

1. Create a job.
2. Create a starter queue and add jobs to that queue.
3. Activate the launch agent. 

When you add a job to your queue the launch agent will automatically create a new run on the compute resource configured for that queue. 

:::note
As a practitioner, you normally only need to complete the following steps to use W&B Launch:

1. Create a job.
2. Add jobs to a queue created by an MLOps Team.
:::


## Satisfy prerequisites
Before you get started, ensure you enable the W&B Launch UI and install Docker. See the Docker documentation for more information on how to install Docker. Make sure Docker is running on your local machine.

1. Navigate to your settings page at [https://wandb.ai/settings](https://wandb.ai/settings) .
2. Scroll to the **Beta Features** sections. Toggle the W&B Launch.

## Create a job
Create a W&B Run similarly to how you normally would create a job. Use `wandb.init()` to start a new run. Use `wandb.log()` to log metrics and specify `run.log_code()` to log code and create a code artifact. W&B will create a job when you create a code artifact with `run.log_code()`. 

Copy and paste the proceeding Python code example as a Python script called `train.py`.

```python
# train.py

import random
import wandb


def run_training_run(epochs, lr):
    print(f"Training for {epochs} epochs with learning rate {lr}")

    run = wandb.init(
        # Set the project where this run will be logged
        project="job_example",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": epochs,
        })

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # simulating a training run
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, acc={acc}, loss={loss}")
        wandb.log({"acc": acc, "loss": loss})

    run.log_code() # log code so we can reuse this experiment


run_training_run(epochs=10, lr=0.01)
```
Navigate to your local machine’s terminal and execute the Python script locally on your machine.
```python
python train.py
```

This creates a W&B Run. In addition, W&B automatically creates a job for you because you logged your code with `run.log_code()`. 

Information about the W&B Run, such as metrics logged, can be found by copying and pasting the W&B Run URL that is printed from the terminal. The output and URL displayed in your terminal will look similar but differ.

![](/images/launch/sample_run_terminal_view.png)

:::info
There is more than one way to create a W&B Job. For more information, see the [Create a job](./create-job.md) page.
:::

## Add the job to your queue
Add the job you created in the previous step to a queue to execute a run locally on your machine. All jobs added to a given queue automatically posses the same resource type(local, dev container, cluster, etc.) and parameter resource arguments.

:::info
In this example we demonstrate how to create a “Starter queue”. This starter queue is designed for local testing and demonstrative purposes only. 
:::

The following procedure demonstrates how to create a starter queue and how to add a job to that queue:

1. Navigate to your W&B Project Page. 
2. Select the **Jobs** icon on the left panel:

![](/images/launch/project_jobs_tab_gs.png)

3. The **Jobs** page displays a list of W&B Jobs that were created from previously executed W&B Runs. 

![](/images/launch/view_jobs.png)

4. Select the **Launch** button next to the name of the Job name. A modal will appear on the right side of the page.
5. Within the modal select the:
    * Job version you want to add to your queue from the **Job version** dropdown. In this example we only have one version, so we select `v0`.
    * Select the **Paste from…** button to automatically propagate hyperparameters used from a specific W&B Run. In the following image, we have two runs to choose from.

![](/images/launch/create_starter_queue_gs.png)

6. Next, select **Starter queue** from the **Queue** dropdown to create a queue.
7. Select the **Launch now** button. 

:::note
If you do not have an active agent running in the background, you will see a red warning banner at the top of the Launch application. The next section describes how to activate an agent.
:::

## Activate a launch agent
Launch queues require an active agent. To start an agent for a local process you will need to execute a CLI command in your local machine’s terminal. 

1. Navigate to the Launch queue’s workspace.
2. Select the **Add an agent** button.
3. A modal will appear with a W&B CLI command. Copy this and paste this command into your terminal.

![](/images/launch/activate_starter_queue_agent.png)

In general, the command to activate an agent takes the form of:

```bash
wandb launch-agent -e <entity-name> -q <queue-name>
```

Within your terminal, you will see the agent begin to poll for queues. Your terminal should look similar to the following image:

![](/images/launch/terminal_gs.png)

That’s it! Navigate to your Launch workspace or your terminal to see the status of your launch jbo. Jobs are executed in first in, first out order (FIFO). All jobs pushed to the queue will use the same resource type and resource arguments. 

## Optional - View your queue and launched jobs

View the status of your queue.

1. Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home) 
2. Select **Launch** within the **Applications** section of the left sidebar.
3. Filter queues by entity. Select the **All entities** dropdown and select the entity to filter with.

For example, in the following image we see the **Starter queue** we created in this guide. 

![](/images/launch/launch_queues_all.png)

View fine-grained details of the job such as run created by a job, the full name of your job, and version metadata.

1. Navigate to your W&B Project.
2. Select the Jobs icon on the left sidebar.
3. The **Jobs** page will appear. In it, you can view all of the jobs created in that project.

![](/images/launch/view_jobs.png)

For example, in the preceding image we have two jobs listed. One is called `job-https__github.com` and the other is called `job-source-launch_demo-canonical_job_example.py`.

Select a job from the **Job ID** column to learn more about that job. A new page with a list of runs created by the job, along with job and version details will appear. 

![](/images/launch/runs_in_job.png)
