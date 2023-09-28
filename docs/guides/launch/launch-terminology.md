---
displayed_sidebar: default
---

# Terms and concepts
With W&B Launch, you enqueue [jobs](#TBD) onto [queues](#TBD). Jobs are python scripts [instrumented with W&B](#TBD). Queues hold a list of jobs to run on a [target resource](#TBD). [Agents](#TBD) pull jobs from queues and execute the jobs on target resources. Launch jobs are tracked similarly to other W&B [runs](#TBD).


### Launch job
A job is a specific type of [W&B Artifact](#TBD) that represents work to be done.  Job definitions include:

- Python code and other file assets, including at least one runnable entrypoint.
- Information about the input (config parameter) and output (metrics logged).
- Information about the environment. (e.g., `requirements.txt`, base `Dockerfile`).


There are three main kinds of job definitions:


| Job types | Definition | How to run this job type | 
| ---------- | --------- | -------------- |
|Artifact-based (or code-based) jobs| Code and other assets are saved as a W&B artifact.| To run artifact-based jobs, Launch agent must be configured with a [builder](#TBD). |
|Git-based jobs|  Code and other assets are cloned from a certain commit, branch, or tag in a git repository. | To run git-based jobs, Launch agent must be configured with a [builder](#TBD) and [git repo credentials](#TBD). |
|Image-based jobs|Code and other assets are baked into a Docker image. | To run image-based jobs, Launch agent might need to be configured with [image repository credentials](#TBD). | 


:::tip
Launch jobs are created automatically when you track a run[LINK] with W&B. You can manually specify what type of launch job is created with the W&B CLI's `wandb job create` [command](#TBD).  See [these docs](#TBD) for more information on how to create launch jobs.
:::

Find jobs you created in the W&B App under the `Jobs` tab of your project workspace.  From there, jobs can be configured and sent to a [Launch queue](#TBD) to be executed on a variety of [target resources](#TBD).

### Launch queue
Launch *queues* are ordered lists of jobs to execute on a specific target resource.  Launch queues are first-in, first-out. (FIFO).  There is no practical limit to the number of queues you can have, but a good guideline is one queue per target resource.  Jobs can be enqueued with the W&B App UI, W&B CLI or Python SDK.  Then, one or more Launch agents can be configured to pull items from the queue and execute them on the queue's target resource.

### Target resources
The compute environment that a Launch queue is configured to execute jobs on is called the *target resource*.

W&B Launch supports the following target resources:

- [Docker](#TBD)
- [Kubernetes](#TBD)
- [AWS SageMaker](#TBD)
- [GCP Vertex](#TBD)

Each target resource accepts a different set of configuration parameters called *resource configurations*. Resource configurations take on default values defined by each Launch queue, but can be overridden independently by each job.  See the documentation for each target resource for more details.

### Launch agent
Launch agents are lightweight, persistent programs that periodically check Launch queues for jobs to execute.  When a launch agent receives a job, it first builds or pulls the image from the job definition then runs it on the target resource.

One agent may poll multiple queues, however the agent must be configured properly to support all of the backing target resources for each queue it is polling.  [LINK]

### Launch agent environment
The agent environment is the environment where a launch agent is running, polling for jobs. [LINK]

:::info
The agent's runtime environment is independent of a queue's target resource.  In other words, agents can be deployed anywhere as long as they are configured sufficiently to access the required target resources.
:::