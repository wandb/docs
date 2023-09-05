---
displayed_sidebar: default
---

# Terms and concepts

W&B Launch has four main concepts: Jobs, Queues, Target Resources and Agents.

When you use W&B Launch, you enqueue [jobs](#TBD) onto [queues](#TBD). Queues run on a [target resource](#TBD). [Agents](#TBD) poll queues and execute jobs on target resource based on the content's of the queue's configuration.


* Launch job
* Launch queue
* Target resource
* Launch agent


#### Launch job
A job is a specific type of [W&B Artifact](#TBD) that represents some work to be done.  Job definitions include:

- Python code and other file assets, including at least one runnable entrypoint.
- Information about inputs (config parameters, used artifacts) and outputs. (metrics, created artifacts)
- Information about the environment. (e.g., `requirements.txt`, base `Dockerfile`).

All jobs have some important constraints and limitations to keep in mind:

- The python script must use the `wandb` library and call `wandb.init(...)` at some point.
- TBD...

There are three main kinds of job definitions:


| Job types | Definition | How to run job | 
| ---------- | --------- | -------------- |
|Artifact-based (or code-based) jobs| Code and other assets are saved as a W&B artifact.| To run artifact-based jobs, Launch agent must be configured with a [builder](#TBD). |
|Git-based jobs|  Code and other assets are cloned from a certain commit, branch, or tag in a git repository. | To run git-based jobs, Launch agent must be configured with a [builder](#TBD) and [git repo credentials](#TBD). |
|Image-based jobs|Code and other assets are baked into a Docker image. | To run image-based jobs, Launch agent might need to be configured with [image repository credentials](#TBD). | 


:::tip
Artifact-based jobs are created automatically when you track a run[LINK] with W&B.  Jobs of all kinds can be created manually with the W&B CLI's `wandb job create` [command](#TBD).  See [these docs](#TBD) for how to create git- or image-based jobs.
:::

Once a job is defined, it can be found in the W&B app under the `Jobs` tab of your project workspace.  From there, jobs can be configured and sent to a [Launch queue](#TBD) to be executed on a variety of [target resources](#TBD).

#### Launch queue
Launch *queues* are ordered lists of jobs to execute on a specific target resource.  Launch queues are first-in, first-out. (FIFO).  There is no practical limit to the number of queues you can have, but a good guideline is one queue per target resource.  Jobs can be enqueued with the W&B App UI, W&B CLI or Python SDK.  Then, one or more Launch agents can be configured to pull items from the queue and execute them on the queue's target resource.

#### Target resources
The compute environment that a Launch queue is configured to execute jobs on is called the *target resource*.

W&B Launch supports the following target resources:

- [Docker](#TBD)
- [Kubernetes](#TBD)
- [AWS SageMaker](#TBD)
- [GCP Vertex](#TBD)

Each target resource accepts a different set of configuration parameters.  So-called "resource configurations" take on default values defined by each Launch queue, but can be overridden independently by each job.  See the documentation for each target resource for more details.

#### Launch agent
Launch agents are lightweight, persistent programs that periodically check Launch queues for jobs to execute.  When a Launch agent receives a job, it first creates or pulls the image from the job definition then runs it on the target resource.

One agent may poll multiple queues, however the agent must be configured properly to support all of the backing target resources for each queue it is polling.  

:::info
The agent's runtime environment is independent of the target resources.  In other words, agents may be deployed anywhere as long as they are configured sufficiently to access the required resources.
:::