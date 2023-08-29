---
description: Getting started guide for W&B Launch.
displayed_sidebar: default
---

# Walkthrough

This guide will walk you through how to setup of the fundamental components W&B launch:  **launch jobs**, **launch queue**, and **launch agents**. By the end of this walkthrough you will:

1. Create a launch job that trains a neural network.
2. Create a launch queue that is used to submit jobs for execution on your local machine.
3. Create a launch agent that polls the queue and executes launch with Docker.

:::note
The steps outlined on this page are designed to run on your local machine with Docker.
:::

## Before you get started

Before you get started, ensure you have satisfied the following prerequisites:
1. Install W&B Python SDK version 0.14.0 or higher:
    ```bash
    pip install wandb>=0.14.0
    ```
2. Sign up for a free account at https://wandb.ai/site and then login to your wandb account. 
3. Install Docker. See the [Docker documentation](https://docs.docker.com/get-docker/) for more information on how to install Docker, and make sure the docker daemon is running on your machine.

## Create a launch job

The following code creates a launch job from a W&B [run](../../ref/python/run.md). Copy the following Python code to your machine in a file named `train.py`.

```python title="train.py"
import wandb

config = {
    "epochs": 10
}

with wandb.init(config=config, project="launch-quickstart") as run:
    config = wandb.config
    for epoch in range(1, config.epochs):
        loss = config.epochs / epoch
        accuracy = (1 + (epoch / config.epochs))/2
        wandb.log({
            "loss": loss, 
            "accuracy": accuracy, 
            "epoch": epoch})
    
    # highlight-next-line
    wandb.run.log_code()  

```

In this example, we log our code during a W&B run with the `log_code()`[INSERT] (see the highlighted portion of the code example).

:::tip
There are different ways to create launch job. For more information, see the Create and deploy jobs section.[LINK]
:::

Execute the Python script and let the script run until it completes:


```bash
python train.py
```


## Add your launch job to a queue

<!-- ![](/images/launch/simple-job.png) -->

1. Navigate to your W&B project. 
2. Select the Jobs tab on the left panel (thunderbolt icon).
3. Hover your mouse next the name of the job you created and select the **Launch** button.
4. A drawer will slide from the right side of your screen. Select the following:
    1. **Job version**: the version of the job to launch.  Since we only have one version, select the default **@latest** version.
    2. **Overrides**: new values for the launch job's inputs. Our run had one value in the `wandb.config`: `epochs`. We can override this value within the overrides field. For this walkthrough, leave the number of epochs as is.
    3. **Queue**: the queue to launch the run on. From the dropdown, select **Create a 'Starter' queue**.

![](/images/launch/starter-launch.gif)
5. Once you have configured your job as desired, click the **Launch now** button at the bottom of the drawer to enqueue your launch job.



:::tip
The contents of your launch queue configuration will vary depending on the queue's target resource[LINK]. For more information about launch queues, see the [INSERT].
:::


## Start a launch agent
To execute the launch job you created in previous steps, you will need a launch agent that polls your launch queue. Follow these steps to create and start a launch agent:

1. From [wandb.ai/launch](https://wandb.ai/launch) navigate to the page for your launch queue.
2. Click the 
3. A modal will appear with a W&B CLI command. Copy this and paste this command into your terminal.

![](/images/launch/activate_starter_queue_agent.png)

In general, the command to start a launch agent is:

```bash
wandb launch-agent -e <entity-name> -q <queue-name>
```

Within your terminal, you will see the agent begin to poll for queues. 

:::tip
Launch agents can poll for queues in non-local environments such as a Kubernetes cluster. For more information, see [LINK].
:::


## View your launch job

Navigate to your new **launch-quickstart** project in your W&B account and open the jobs tab from the navigation on the left side of the screen.

![](/images/launch/jobs-tab.png)

The **Jobs** page displays a list of W&B Jobs that were created from previously executed runs. You should see a job named **job-source-launch-quickstart-train.py:v0**. Click on your launch job to view source code dependencies and a list of runs that were created by the launch job.

:::tip
You can edit the name of the job from the jobs page if you would like to make the job a bit more memorable. For more information, see [LINK]. 
:::
