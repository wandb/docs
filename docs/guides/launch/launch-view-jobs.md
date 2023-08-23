---
displayed_sidebar: default
---

# View launch jobs

The following page describes how to view information, such as the status of a job, of launch jobs that were added to a queue.

View jobs added to a queue with the W&B App.

1. Navigate to the W&B App at https://wandb.ai/home.
2. Select **Launch** within the **Applications** section of the left sidebar.
3. Select the **All entities** dropdown and select the entity to filter with.
4. Expand the collapsible queue UI from the Launch Application page to view jobs added to a specific queue.

:::info
A run is created when the launch agent executes a launch job. In other words, each run listed corresponds to a specific job that was added to that queue.
:::

For example, the following image shows two runs that were created from a job called `job-source-launch_demo-canonical`. The job was added to a queue called `Start queue`. The first run listed in the queue called `resilient-snowball` and the second run listed is called `earthy-energy-165`.


![](/images/launch/launch_jobs_status.png)

Within the W&B App UI you can find additional information about runs created from launch jobs such as the:
   - **Run**: The name of the W&B run assigned to that job.
   - **Job ID**: The name of the job. See Job naming conventions[LINK] page for information on the default naming assigned to a job.
   - **Project**: The name of the project the run belongs to.
   - **Status**: The status of the queued run. 
   - **Author**: The W&B entity that created the run.
   - **Creation date**: The timestamp when the queue was created.
   - **Start time**: The timestamp when the job started.
   - **Duration**: Time, in seconds, it took to complete the job’s run.

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
| **Failed** | The run ended with a non-zero exit code. |
| **Finished** | The job completed successfully. |
