---
description: Discover how to create a queue for W&B Launch.
displayed_sidebar: default
---

# Create a queue

**Launch queues** are first-in, first-out (FIFO) queues where users can configure and submit their jobs to be executed by any polling launch agents. Each item in a launch queue consists of a job and settings for the parameters of that job. Launch queues are owned by either a W&B or W&B Team. Follow this guide to learn how to create and configure a launch queue.

## Create a queue

Navigate to [wandb.ai/launch](https://wandb.ai/launch). Click **create queue** button in the top right of the screen to start creating a new launch queue.

When you click the button, a drawer will slide from the right side of your screen and present you with some options for your new queue:

* **Entity**: the owner of the queue, which can either be your W&B account or any W&B team you are a member of. For this tutorial, we recommend setting up a personal queue.
* **Queue name**: the name of the queue. Make this whatever you want!
* **Resource**: the compute resource type for jobs in this queue.
* **Configuration**: default resource configurations for all jobs placed on this queue. Used to configure the compute resources that any job in this queue will run on. These defaults can be overridden when you enqueue a job.

![](/images/launch/create-queue.gif)


## View queues
View queues associated with a W&B entity with the W&B App.

1. Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home). 
2. Select **Launch** within the **Applications** section of the left sidebar.
3. Select the **All entities** dropdown and select the entity to filter with.

A list of queues will appear in the Launch workspace. From left to right, each queue listed shows the name of the queue, the entity the queue belongs to, the compute resource configured for that queue and the state of the queue. 

For example, in the following image, we have a queue called **Starter queue**. It belongs the the wandb entity. The queue is configured to use Docker. And the queue is in an **inactive (Not running)** state.

![](/images/launch/launch_queues_all.png)

A queue is either **Active** or **Not running** (inactive). 
* **Active**: A queue is active if at least one agent is either polling or executing jobs for that queue. 
* **Not running**: A queue is not running if there is no agent polling or executing jobs for that queue.
