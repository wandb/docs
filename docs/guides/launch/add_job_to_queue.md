---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Enqueue jobs

The following page describes how to add launch jobs to a launch queue.

:::info
Ensure that you, or someone on your team, has already configured a launch queue. For more information, see the [LINK].
:::




## Add jobs to your queue

Add jobs to your queue interactively with the W&B App or programmatically with the W&B CLI.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'W&B CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
Add a job to your queue with the W&B App.

1. Navigate to your W&B Project Page.
2. Select the **Jobs** icon on the left panel:

![](/images/launch/project_jobs_tab_gs.png)

3. The **Jobs** page displays a list of W&B launch jobs that were created from previously executed W&B runs. 

![](/images/launch/view_jobs.png)

4. Select the **Launch** button next to the name of the Job name. A modal will appear on the right side of the page.
5. Within the modal select the:
  * **Job version**: the version of the job to launch. Jobs are versioned like any other W&B Artifact. Different versions of the same job will be created if you make modifications to the software dependencies or source code used to run the job. Since we only have one version, we will select the default **@latest** version.
  * **Overrides**: new values for any of jobs inputs. These can be used to change the entrypoint command, arguments, or values in the `wandb.config` of your new run. Our run had one value in the `wandb.config`: `epochs`. We can override this value by in the overrides field. We can also paste values from other runs using this job by clicking the **Paste from...** button.
  * **Queue**: the queue to launch the run on. If you have not created any queues yet, you should have the option to create a **Starter Queue**. This queue will be used to launch runs on your local machine using Docker.

![](/images/launch/create_starter_queue_gs.png)

6. Next, select **Starter queue** from the **Queue** dropdown to create a queue.
7. Select the **Launch now** button. 


  </TabItem>
    <TabItem value="cli">

Use the `wandb launch` command to add jobs to a queue. Create a JSON configuration with hyperparameter overrides. For example, using the script from the [Quickstart](./walkthrough.md) guide, we create a JSON file with the following overrides:

```json title="config.json"
{
    "overrides": {
        "args": [],
        "run_config": {
            "learning_rate": 0,
            "epochs": 0
        },   
        "entry_point": []
    }
}
```

:::note
W&B Launch will use the default parameters if you do not provide a JSON configuration file.
:::

If you want to override the queue configuration, or if your your launch queue does not have a configuration resource defined, you can specify the `resource_args` key in your config.json file. For example, following continuing the example above, your config.json file might look similar to the following:

```json title="config.json"
{
    "overrides": {
        "args": [],
        "run_config": {
            "learning_rate": 0,
            "epochs": 0
        },
        "entry_point": []
    },
    "resource_args": {
         "<resource-type>" : {
             "<key>": "<value>"
          }
    }
}
```

Replace values within the `<>` with your own values.



Provide the name of the queue for the `queue`(`-q`) flag, the name of the job for the `job`(`-j`) flag, and the path to the configuration file for the `config`(`-c`) flag.

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
If you work within a W&B Team, we suggest you specify the `entity` flag (`-e`) to indicate which entity the queue will use.

  </TabItem>
</Tabs>


