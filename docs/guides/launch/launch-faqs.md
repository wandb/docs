---
description: Answers to frequently asked question about W&B Launch.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch FAQs

<head>
  <title>Frequently Asked Questions About Launch</title>
</head>


## Getting Started


### I don’t want you to build a container for me, can I still use Launch?
  
Yes! You can launch a pre-built docker image by running

    wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
  
  This will build a job when run.
  
  Alternatively, you can make a launch job based on a docker image: see [here](https://docs.wandb.ai/guides/launch/create-job)

### Are there any best practices for using Launch effectively?

  1. Create your queue before you start your agent, so that you can set your agent to point to it easily.  If you don’t do this, your agent will give errors and not work until you add a queue.
  2. Create a W&B service account to start up the agent, so that it's not tied to an individual user account.
  3. Use `wandb.config` to read and write your hyperparameters, as opposed to argsparse, so that they can be overwritten when re-running a job.  Check out [this guide](https://docs.wandb.ai/guides/launch/create-job#making-your-code-job-friendly) if you use argsparse.

### I don’t like clicking- can I use Launch without going through the UI?
  
  Yes! The standard `wandb` CLI includes a `launch` subcommand that you can use to launch your jobs. For more info, try running

    wandb launch --help

### Can Launch automatically provision (and spin down) compute resources for me in the target environment?

No--Launch uses existing resources you have created.  That said, our Solution Architects are happy to work with you to configure your underlying Kubernetes infrastructure to faciliate retries, autoscaling, and use of spot instance node pools.  Reach out to support@wandb.com or in your shared Slack channel.

### `wandb launch -d` uploading a whole docker artifact and is not pulling from a registry? 

  No, the command `wandb launch -d` won't upload to a registry, you'll have to do that yourself.  The process is for you to build an image, push it to a registry, then the agent will spin up a job pointing to that container. For Kubernetes, the k8s cluster pods will need access to the registry you are pushing to. The workflow looks like:

    docker build -t <repo-url>:<tag> .
    docker push <repo-url>:<tag>
    wandb launch -d <repo-url>:<tag>


## Permissions and Resources

### How do I control who can push to a queue?

Queues are scoped to a team of users—you set the owning entity when you create the queue.  So to restrict access, you can change the team membership.

### What permissions does the agent require in Kubernetes?
  1. [https://docs.wandb.ai/guides/launch/kubernetes](https://docs.wandb.ai/guides/launch/kubernetes)
  2. “The following kubernetes manifest will create a role named
  `wandb-launch-agent` in the`wandb`namespace. This role will allow the agent to create pods, configmaps, secrets, and pods/log in the `wandb` namespace. The `wandb-cluster-role` will allow the agent to create pods, pods/log, secrets, jobs, and jobs/status in any namespace of your choice.”*

### Does Launch support parallelization?  How can I limit the resources consumed by a job?
   
  An individual Launch agent is configured with a `max_jobs` parameter that determines how many jobs that agent can be running simultaneously. Additionally, you may point as many agents as you want at a particular queue, so long as those agents are themselves connected to infrastructure that they can launch into.
   
  You can limit the CPU/GPU, memory, etc. requirements at either the queue or job run level, in the resource config. For more information about setting up queues with resource limits on kubernetes see [here](https://docs.wandb.ai/guides/launch/kubernetes#queue-configuration). 
   
  For sweeps, in the SDK you can add a block to the config
    
    scheduler:
    
    num_workers: 4

### How can admins restrict what ML engineers have access to modify? For example, changing an image tag may be fine but other job settings may not be.
  
  Right now, the only permission restriction is that only team admins can create queues.  We are anticipating (a) expanding that to include also *editing* queue configs and (b) to allow whitelisting of certain config parameters to be editable by non-admins.  For example the image tag or the memory requirements.