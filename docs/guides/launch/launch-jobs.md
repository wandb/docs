---
description: Discover how to launch your jobs.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch a run

Use W&B launch to queue jobs for execution by a launch agent. The following guide demonstrates how to submit runs to a queue.

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

Use the `wandb launch` command to add jobs to a queue. Create a JSON configuration with hyperparameter overrides. For example, using the script from the [Quickstart](./quickstart.md) guide, we create a JSON file with the following overrides:

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
   - **Job ID**: The name of the job. See [Job naming conventions](create-job.md#job-naming-conventions) page for information on the default naming assigned to a job.
   - **Project**: The name of the project the run belongs to.
   - **Status**: The status of the queued run. 
   - **Author**: The W&B entity that created the run.
   - **Creation date**: The timestamp when the queue was created.
   - **Start time**: The timestamp when the job started.
   - **Duration**: Time, in seconds, it took to complete the job’s run.


## Status of queued runs
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
