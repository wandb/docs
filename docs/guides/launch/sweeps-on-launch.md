---
description: Discover how to automate hyperparamter sweeps on launch.
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sweeps on Launch

Launch jobs directly from within W&B locally on your machine or to the compute provider of your choice. Reproduce runs or sweeps directly from the W&B User Interface to launch new experiments and compare results.

There are two methods of creating sweeps on launch. First first harnesses the existing Wandb sweep scheduling engine that powers standard sweeps. The familiar `bayes`, `grid`, and `random` methods are available. It is recommended for first-time sweeps on launch users to use this path, and the steps to do so are below in the "basic" section. The second method is more extendible, configuring the sweep scheduler to run as a job itself. This enables full customization, and should be used by users who find standard wandb scheduling lacking.

## Create a basic sweep
Create W&B Sweeps with Launch. You can create a sweep interactively with the W&B App or programmatically with the W&B CLI. For advanced configurations of Launch sweeps, including the ability to customize the scheduler, use the CLI. 

:::info
Before you create a sweep with W&B Launch, ensure that you create a job first. Inspect that the run you want to create a job from has a code artifact. See the [Create a Job](./create-job.md) page for more information. 
:::


<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'CLI', value: 'cli'},
    {label: 'Resume', value: 'existing'}
  ]}>
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
  <TabItem value="cli">

Programmatically create a W&B Sweep with Launch with the W&B CLI.

1. Create a Sweep configuration
2. Specify the full job name within you sweep configuration
3. Initialize a sweep agent.

:::info
Steps 1 and 3 are the same steps you normally take when you create a W&B Sweep. With the exception that you need to specify the name of the job within your sweep YAML configuration file. 
:::

For example, in the following code snippet, we specify `'wandb/jobs/Hello World 2:latest'` for the job value:

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: sweep examples using launch jobs

method: bayes
metric:
  goal: minimize
  name: loss_metric
parameters:
  learning_rate:
    max: 0.02
    min: 0
    distribution: uniform
  epochs:
    max: 20
    min: 0
    distribution: int_uniform

# Optional/advanced scheduler params:
scheduler:
   num_workers: 1  # concurrent sweep runs
   docker_image: <base image for the scheduler>
   resource: <ie. kubernetes, local-container...>
   resource_args:  # resource arguments passed to runs
      env: 
         - WANDB_API_KEY

# Optional Launch Params
launch: 
   registry: <registry for image pulling>
```

For information on how to create a sweep configuration, see the [Define sweep configuration](../sweeps/define-sweep-configuration.md) page.

4. Next, initialize a sweep. Provide the path to your config file, the name of your job queue, your W&B entity, and the name of the project for the queue, entity, and project flags, respectively.

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

For more information on W&B Sweeps, see the [Tune Hyperparameters](../sweeps/intro.md) chapter.


</TabItem>
<TabItem value="existing">

  It is also possible to resume a launch-sweep from a previously launched sweep. Although hyperparameters and the training job cannot be changed, scheduler-specific parameters can be, as well as the queue it is pushed to.

:::info
If the initial sweep used a training job with an alias like 'latest', resuming can lead to different results if the latest job version has been changed since the last run.
:::

  1. Identify the sweep name/id for a previously run launch-sweep (ex. `hhd16935`)
  2. If changing scheduler parameters, construct an updated config.
  3. Run: 

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```
  </TabItem>
</Tabs>


## Create an advanced sweep

Sweeps on Launch are far more customizable than standard sweeps. The sweep scheduling mechanism can be entirely replaced with a job! You can use pre-made schedulers or implement your own custom scheduler job. Examples of what is possible with custom sweep scheduler jobs are available in the [wandb/launch-jobs](https://github.com/wandb/launch-jobs) repo under `jobs/sweep_schedulers`. This guide shows how to use the publicly available **Wandb Scheduler Job**, as well demonstrates a process for creating custom sweep scheduler jobs. 

What is a sweep scheduler job? Just like any other training job, launch will take a scheduler job and execute it in the environment of choice. The sweep scheduler starts, turning a user provided hyperparameter configuration into many different sweep runs. Rather than starting runs in a sub-process like standard sweeps, the scheduler packages them up and launches them onto the same queue it is running on. Then, the scheduler is responsible for polling on the runs, feeding that information back into its optimization algorithm, and determining which combinations of parameters to use next.


:::info
Using scheduler jobs requires wandb cli version >= `0.15.4`
:::

<Tabs
  defaultValue="wandb-scheduler"
  values={[
    {label: 'Wandb Scheduler', value: 'wandb-scheduler'},
    {label: 'Custom Scheduler', value: 'custom-scheduler'},
    {label: 'Optuna Scheduler', value: 'optuna-scheduler'},
  ]}>
    <TabItem value="wandb-scheduler">

  Create a launch sweep using the wandb sweep scheduling logic, as a job.
  
  1. Identify the 'Wandb Sweep Scheduler' job in the public wandb/jobs project, or use the job name:
  `'wandb/jobs/Wandb Sweep Scheduler:latest'`
  2. Construct a configuration yaml with an additional `scheduler` block that includes a `job` key pointing to this name, example below.
  3. Use the `wandb launch-sweep` command with the new config.


Example config:
```yaml
# launch-sweep-config.yaml  
description: Launch sweep config using a scheduler job
scheduler:
  job: 'wandb/jobs/Wandb Sweep Scheduler:latest'
  num_workers: 8  # allows 8 concurrent sweep runs

# training/tuning job that the sweep runs will execute
job: 'wandb/jobs/Hello World 2:latest'
method: grid
parameters:
  param1:
    min: 0
    max: 10
```

  </TabItem>
  <TabItem value="custom-scheduler">

  Custom schedulers can be created by creating a scheduler-job. For the purposes of this guide we will be modifying the `WandbScheduler` to provide more logging. 

  1. Clone the `wandb/launch-jobs` repo (specifically: `wandb/launch-jobs/jobs/sweep_schedulers`)
  2. Now, we can modify the `wandb_scheduler.py` to achieve our desired increased logging. Example: Add logging to the function `_poll`. This is called once ever polling cycle (configurable timing), before we launch new sweep runs. 
  3. Run the modified file to create a job, with: `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. Identify the name of the job created, either in the UI or in the output of the previous call, which will be a code-artifact job (unless otherwise specified).
  5. Now create a sweep configuration where the scheduler points to your new job!

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'

...
```

  </TabItem>
  <TabItem value="optuna-scheduler">

  Coming soon
  </TabItem>
</Tabs>
