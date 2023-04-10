---
description: Discover how to create a queue for W&B Launch.
---

# Create a queue

**Launch queues** re first-in, first-out (FIFO) queues where users can configure and submit their jobs to a particular compute resource. Each item in a launch queue consists of a job and settings for the parameters of that job. Follow this guide to learn how to create and configure a launch queue.

## Create a queue

Navigate to [wandb.ai/launch](https://wandb.ai/launch). Click **create queue** button in the top right of the screen to start creating a new launch queue.

When you click the button, a drawer will slide from the right side of your screen and present you with some options for your new queue:

* **Entity**: the owner of the queue, which can either be your W&B account or any W&B team you are a member of. For this demo, we recommend setting up a personal queue.
* **Queue name**: the name of the queue. Make this whatever you want!
* **Resource**: the execution platform for jobs in this queue.
* **Configuration**: default resource configurations for all jobs placed on this queue. Used to specify arguments for the resource when run. These defaults can be overridden when you enqueue a job.

![](/images/launch/create-queue.gif)
