import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch jobs
[INSERT]


## Recreate a run
Add your W&B Job to a queue to execute a W&B Run. All jobs pushed to a queue automatically posses the same resource type(local, dev container, cluster, etc.) and parameter resource arguments.

Interactively recreate a W&B Run with the W&B App or programmatically with the W&B CLI.

<Tabs
  defaultValue="app"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="cli">
Follow the procedure outlined below to recreate a run with the CLI:

1. Find the name of the run you want to reproduce and its associated job name. See the [INSERT] page for information on how to find the job name.
2. Navigate to your terminal and type the following:

```bash    
wandb launch -j <job-name> -c path/to/config_file.j
```    
This will default to running on the `Docker runner` on the same machine you intend on running on.

:::info
You need to specify the `run.config` in your config files `overrides.run_config` keys.
:::

The following code snippet demonstrates an example config file:

```python
{
   "overrides": {
      "args": [],
      "run_config": {
	      "lr": <value>,
        "batch_size": <value>
      }
   }
}
```

  </TabItem>
  <TabItem value="app">
The following procedure demonstrates how to add a job to a queue interactively with the W&B App.

:::info
In this example we demonstrate how add a job to a “Starter queue”. This starter queue is designed to be used for local testing and demonstrative purposes only. 

For more information on how to create queues that utilize other cloud compute resources such as Kubernetes, see [FUTURE LINK TO DOCS SECTION]. 
:::

1. Navigate to your project page on the W&B App.
2. Select the **Jobs Tab**.

![](/images/launch/project_jobs_tab_gs.png)

3. The Jobs Page displays a list of W&B Jobs that were created from previously executed W&B Runs. Find information about the jobs such as the:

* **Job ID**: Unique Job ID name. For more information about Job naming conventions, see Job Naming Conventions[LINK].
* **Versions**: The number of job versions.
* **Runs**: The number of W&B Runs created by the associated job.
* **Creation date**: The creation date of the job.
* **Last run**: The timestamp of the last run created by the job.

![](/images/launch/view_jobs.png)

4. Select the **Launch** button next to the job name. A UI modal will appear.
5. Within the modal select the:
* Job version you want to add to your Queue from the **Job version** dropdown. In this example we only have one version, so we select `v0`. 
* Select the **Paste from…** button to automatically propagate hyperparameters used from a specific W&B Run. In the following image, we have two Runs to choose from.

![](/images/launch/create_starter_queue_gs.png)
6. From the Queue dropdown, select Starter queue. 

</TabItem>
</Tabs>



## Create a sweep