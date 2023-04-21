---
description: Discover how to automate hyperparamter sweeps on launch.
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sweeps on Launch

Launch jobs directly from within W&B locally on your machine or to the compute provider of your choice. Reproduce runs or sweeps directly from the W&B User Interface to launch new experiments and compare results.

## Create a sweep
Create W&B Sweeps with Launch. You can create a sweep interactively with the W&B App or programmatically with the W&B CLI.

:::info
Before you create a sweep with W&B Launch, ensure that you create a job first. Inspect that the run you want to create a job from has a code artifact. See the [Create a Job](./create-job.md) page for more information. 
:::


<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'CLI', value: 'cli'},
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

For information on how to create a sweep configuration, see the [Define sweep configuration](../sweeps/define-sweep-configuration.md) page.

4. Next, initialize a sweep. Provide the path to your config file, the name of your job queue, your W&B entity, and the name of the project for the queue, entity, and project flags, respectively.

```bash
wandb sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

For more information on W&B Sweeps, see the [Tune Hyperparameters](../sweeps/intro.md) chapter.


  </TabItem>
</Tabs>
