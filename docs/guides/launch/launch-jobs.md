import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch jobs
[INSERT]


## Recreate a run
Add your W&B Job to a queue to execute a W&B Run. All jobs pushed to a queue automatically posses the same compute resource type and parameter resource arguments. You can alter the hyperparameters to use when you add a job to a queue.

Interactively recreate a W&B Run with the W&B App or programmatically with the CLI.

<Tabs
  defaultValue="cli"
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
2. Select the **Jobs** tab.

![](/images/launch/project_jobs_tab_gs.png)

3. The **Jobs Page** displays a list of W&B Jobs that were created from previously executed W&B Runs. Find information about the jobs such as the:

* **Job ID**: Unique Job ID name. For more information about Job naming conventions, see Job Naming Conventions[LINK].
* **Versions**: The number of job versions.
* **Runs**: The number of W&B Runs created by the associated job.
* **Creation date**: The creation date of the job.
* **Last run**: The timestamp of the last run created by the job.

![](/images/launch/view_jobs.png)

4. Select the **Launch** button next to the job name. A UI modal will appear.
5. Within the modal select the:
* Job version you want to add to your queue from the **Job version** dropdown. In this example we only have one version, so we select `v0`. 
* Select the **Paste from…** button to automatically propagate hyperparameters used from a specific W&B Run. In the following image, we have two runs to choose from.

![](/images/launch/create_starter_queue_gs.png)
6. From the **Queue** dropdown, select **Starter queue**.  
7. Select the **Launch now** button. 

:::info
The described example use a queue called **Starter queue**. This is a queue that W&B created for demonstrative purposes only. In addition to adding the job to this queue, W&B will also create this queue for you. The **Starter queue** uses a Docker image to run the job locally on your machine. 

Note that W&B expects that you create a queue before you can add jobs to it. For more information on how to create a queue, see the Prerequisites[LINK].
:::


</TabItem>
</Tabs>



## Create a sweep
Create W&B Sweeps with Launch. You can create a sweep interactively with the W&B App or programmatically with the W&B CLI.

:::info
Before you create a sweep with W&B Launch, ensure that you create a job first. Inspect that the run you want to create a job from has a code artifact. See the Create a Job[LINK] page for more information. 
:::


<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="cli">
Programmatically create a W&B Sweep with Launch with the W&B CLI.

1. Create a Sweep configuration
2. Specify the full job name within you sweep configuration
3. Initialize a sweep agent.

:::info
Steps 1 and 3 are the same steps you normally take when you create a W&B Sweep. With the exception that you need to specify the name of the job within your sweep YAML configuration file. 
:::

For example, in the following code snippet, we specify `wandb/launch_demo/job-source-launch_demo-canonical_job_example.py:v0` for the job value:

```yaml
#config.yaml

job: wandb/launch_demo/job-source-launch_demo-canonical_job_example.py:v0
description: sweep examples using launch jobs

method: bayes
metric:
  goal: minimize
  name: ""
parameters:
  learning_rate:
    max: 0.02
    min: 0
    distribution: uniform
  epochs:
    max: 20
    min: 0
    distribution: int_uniform
```

For information on how to create a sweep configuration, see [LINK].

4. Next, initialize a sweep. Provide the path to your config file, the name of your job queue, your W&B entity, and the name of the project for the queue, entity, and project flags, respectively.

```bash
wandb sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

For more information on W&B Sweeps, see the Sweeps Guide [LINK].


  </TabItem>
  <TabItem value="app">
Create a sweep interactively with the W&B App.

1. Navigate to you W&B project on the W&B App.  
2. Select the sweeps icon on the left panel (broom image). 
3. Next, select the **Create Sweep** button.
4. Toggle the **Use Launch 🚀 (In Beta)** slider.
5. From the **Job** dropdown menu, select the name of your job and the job version you want to create a sweep from. 
6. Select the queue to add the job to from the **Queue** dropdown menu.
7. Select **Initialize Sweep**.

![](/images/launch/create_sweep_with_launch.png)

  </TabItem>
</Tabs>


## View details of launched jobs
[INSERT]

### View job artifacts
Each W&B Job you create is saved as a W&B Artifact. Select the **Artifacts** icon within your project’s workspace on the W&B App to view a list of job artifacts created in that project.

![](/images/launch/job_artifacts_project_page.png)

Expand the **JOB** menu on the left panel to view a list of job artifacts. For example, in the following image we have two job artifacts called: 
- **job-https___github.com_githubrepo_demo_launch.git_canonical_job_example.py**
- **job-source-launch_demo-canonical_job_example.py**

![](/images/launch/job_artifacts_page.png)

### View details of each job

Navigate to your W&B Project to view fine-grained details of each job such as runs created by a job, the full name of your jobs, and version metadata associated with a project. 

1. Navigate to your W&B project.
2. Select the **Jobs** icon on the left sidebar.
3. A **Jobs** page will appear. In it, you can view all of the jobs created in that project.

![](/images/launch/view_jobs.png)

For example, in the following image we have two job listed:
- **job-https___github.com_githubrepo_demo_launch.git_canonical_job_example.py**
- **job-source-launch_demo-canonical_job_example.py**

Select a job from list to learn more about that job. A new page with a list of runs created by the job, along with job and version details will appear.  This information is contained in three tabs: **Runs**, **Job details**, and **Version details**.

<Tabs
  defaultValue="runs"
  values={[
    {label: 'Runs', value: 'runs'},
    {label: 'Job details', value: 'jobs_details'},
    {label: 'Version details', value: 'version_details'},
  ]}>
  <TabItem value="runs">

Select the name of your job from the list. This will redirect you to a new page with details about each run created by the job such as the:

The Runs tab provides information about each run created by the job such as the:

- **Run**: The names of run.
- **State**: The state of the run.
- **Job version**: The version of the job used.
- **Creator**: Who created the run.
- **Creation date**: The timestamp of when the run
- **Other**: The remaining columns will contain the key-value pairs of the configuration dictionary passed to `wandb.init()`. 

For example, in our demo script, we passed the learning rate (`learning_rate`) and number of epochs (`epochs`) when we initialized a run with `wandb.init()`.

![](/images/launch/runs_in_job.png)


  </TabItem>
  <TabItem value="jobs_details">

The **Job details** provides information about:

* **Description**: An optional description of the job. Select the pencil icon next to this field to add a description.
* **Owner entity**: The entity the job belongs to.
* **Parent project**: The project the job belongs to.
* **Full name**: The full name of your job
* **Creation date**: Creation date of the job.


![](/images/launch/job_id_full_name.png)

  </TabItem>
  <TabItem value="version_details">

Use the **Version details** tab to view specific information about each job version such as the input and output types, and files used for each job version. 

[INSERT image of this]

  </TabItem>
</Tabs>