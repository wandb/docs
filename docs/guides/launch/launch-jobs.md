---
description: Learn how to add jobs to your W&B queue.
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch a run

Use W&B launch to queue up jobs for execution on a particular compute resource. Users submit workloads in the form of jobs to a launch queue, then any agent polling on that queue can pop a workload and execute it according to the configuration and type of the queue, e.g. via Docker, Kubernetes, etc. Queues can be created by and for individual W&B users, or shared across W&B Teams. In either case, queue setup should be done once and then jobs can be added to the queue as needed. In this guide, we will walk through how to add jobs to a launch queue once it has been set up.

## Add jobs to your queue
Add jobs to your queue interactively with the W&B App or programmatically with the CLI.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
Add a job to your queue with the W&B App.

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


  </TabItem>
    <TabItem value="cli">

Use the `wandb launch` command to add jobs to a queue. Create a JSON configuration with hyperparameter overrides. For example, using the script from the [Getting Started](./getting-started.md) guide, we create a JSON file with the following overrides:

```json
// config.json
{
    "args": [],
    "run_config": {
        "learning_rate": 0,
        "epochs": 0
    },
    "entry_point": []
}
```
W&B Launch will use the default parameters if you do not provide a JSON configuration file.


Provide the name of the queue for the `queue`(`-q`) flag, the name of the job for the `job`(`-j`) flag, and the path to the configuration file for the `config`(`-c`) flag.

```bash
wandb launch -j <job> -q <queue-name> -e <entity-name> -c path/to/config.json
```
If you work within a W&B Team, we suggest you specify the `entity` flag (`-e`) to indicate which entity the queue will use.

  </TabItem>
</Tabs>

## View queued jobs
View jobs added to a queue with the W&B App.

1. Navigate to the W&B App at https://wandb.ai/home.
2. Select **Launch** within the **Applications** section of the left sidebar.
3. Select the **All entities** dropdown and select the entity to filter with.
4. Expand the collapsible queue UI from the Launch Application page to view jobs added to a specific queue.

Each run listed corresponds to a job that was was added to that queue. For example, the following image shows there are two jobs listed in a queue called `Starter queue`. One is called `resilient-snowball` and the other is called `earthy-energy-165`:

![](/images/launch/launch_jobs_status.png)

Find additional information about the jobs such as the:
   - **Run**: The name of the W&B Run assigned to that job.
   - **Job ID**: The name of the job. See [Job naming conventions](create-job#job-naming-conventions) page for information on the default naming assigned to a job.
   - **Project**: The name of the project the run belongs to.
   - **Status**: The status of the queued run. 
   - **Author**: The W&B entity that created the run.
   - **Creation date**: The timestamp when the queue was created.
   - **Start time**: The timestamp when the job started.
   - **Duration**: Time, in seconds, it took to complete the job’s run.


## Status of queued runs

| Status | Description |
| --- | --- |
| **-- Idle** | The run is in a queue with no active agents. |
| **Claimed** | The run has been picked up by an agent but has not yet started. |
| **Running** | The run is currently executing. |
| **Killed** | The job was killed by the user. |
| **Failed** | The run ended with a non-zero exit code. |
| **Finished** | The job completed successfully. |
