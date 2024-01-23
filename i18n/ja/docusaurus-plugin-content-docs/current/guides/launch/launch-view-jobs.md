---
displayed_sidebar: ja
---

# View launch jobs

The following page describes how to view information about launch jobs added to queues.

## View jobs

View jobs added to a queue with the W&B App.

1. Navigate to the W&B App at https://wandb.ai/home.
2. Select **Launch** within the **Applications** section of the left sidebar.
3. Select the **All entities** dropdown and select the entity the launch job belongs to.
4. Expand the collapsible UI from the Launch Application page to view a list of jobs added to that specific queue.

:::info
A run is created when the launch agent executes a launch job. In other words, each run listed corresponds to a specific job that was added to that queue.
:::

For example, the following image shows two runs that were created from a job called `job-source-launch_demo-canonical`. The job was added to a queue called `Start queue`. The first run listed in the queue called `resilient-snowball` and the second run listed is called `earthy-energy-165`.


![](/images/launch/launch_jobs_status.png)

Within the W&B App UI you can find additional information about runs created from launch jobs such as the:
   - **Run**: The name of the W&B run assigned to that job.
   - **Job ID**: The name of the job. 
   - **Project**: The name of the project the run belongs to.
   - **Status**: The status of the queued run. 
   - **Author**: The W&B entity that created the run.
   - **Creation date**: The timestamp when the queue was created.
   - **Start time**: The timestamp when the job started.
   - **Duration**: Time, in seconds, it took to complete the job’s run.

## List jobs 
View a list of jobs that exist within a project with the W&B CLI. Use the W&B job list command and provide the name of the project and entity the launch job belongs to the `--project` and `--entity` flags, respectively. 

```bash
 wandb job list --entity your-entity --project project-name
```

## Check the status of a job

The following table defines the status a queued run can have:


| Status | Description |
| --- | --- |
| **Idle** | The run is in a queue with no active agents. |
| **Queued** | The run is in a queue waiting for an agent to process it. |
| **Starting** | The run has been picked up by an agent but has not yet started. |
| **Running** | The run is currently executing. |
| **Killed** | The job was killed by the user. |
| **Crashed** | The run stopped sending data or did not successfully start. |
| **Failed** | The run ended with a non-zero exit code or the run failed to start. |
| **Finished** | The job completed successfully. |


## Automatic run re-queuing on preemption

In some cases, it can be useful to set up jobs to be resumed after they are interrupted.  For example, you might run broad hyperparameter sweeps on spot instances, and want them to pick up again when more spot instances spin up.  Launch can support this configuration on Kubernetes clusters.

If your Kubernetes queue is running a job on a node that’s pre-empted by a scheduler, the job will be automatically added back to the end of the queue so it can resume later. This resumed run will have the same name as the original, and can be followed from the same page in the UI as the original. A job can be automatically re-queued this way up to five times. 

Launch detects whether a pod is preempted by a scheduler by checking if the pod has the condition `DisruptionTarget` with one of the following reasons:

- `EvictionByEvictionAPI`
- `PreemptionByScheduler`
- `TerminationByKubelet`

If your job’s code is structured to allow resuming, it will enable these re-queued runs to pick up where they left off. Otherwise, runs will start from the beginning when they are re-queued. See our guide for [resuming runs](https://docs.wandb.ai/guides/runs/resuming) for more info.   

There is currently no way to opt out of automatic run re-queuing for preempted nodes. However, if you delete a run from the UI or delete the node directly, it will not be re-queued.

Automatic run re-queuing is currently only available on Kubernetes queues; Sagemaker and Vertex are not yet supported.