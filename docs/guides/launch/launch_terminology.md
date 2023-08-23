---
displayed_sidebar: default
---

# Terms and concepts

The following are basic terms and concepts that will help you get started with W&B Launch:

[INSERT bullet points w/ hyperlinks to each term here]


#### Job template
A job is an Artifact that is created automatically when you track a run with W&B. Each job template contains contextual information about the run it is being created from, including the source code, entrypoint, software dependencies, hyperparameters, dataset version, etc.

#### Job
Text.



#### Job config
Text

#### Resources
Text

#### Target resources
The compute environment that a queue is configured to execute jobs on is called the *target resource*.

For example, if a launch job is popped off a Docker queue, the agent will execute the run locally with the `docker run` command. If the job was in a Kubernetes queue, the agent will execute the run on a Kubernetes cluster as a [Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) with the Kubernetes API.

#### Queue
Launch *queues* are first in, first out (FIFO) queues that pop off launch jobs. The launch queue uses the target resource you define in a launch queue configuration to execute the jobs on that queue. 

#### Queue configuration
The configuration of your launch queue. You specify the launch queue when you create a queue. The schema of your launch queue configuration depends on the target compute resource jobs are executed on. 

For example, the queue configuration for an Amazon SageMaker queue target resource will differ from that of a Kubernetes cluster queue target resource.

#### Agent

The agent polls on one or more queues. When the launch agent pops an item from a queue, it will, if necessary, build a container image to execute the run within and then execute that container image on the compute platform targeted by the queue.

These container builds and executions happen asynchronously. 


#### Agent environment
The environment that a launch agent is running in, and polling for launch jobs, is called the *agent environment*. Example agent environments include: locally on your machine or Kubernetes clusters. See the Launch agent environments[LINK] section for more information.

The launch agent environment is independent of a queue's launch target resource.