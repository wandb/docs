---
slug: /guides/integrations/openai
description: How to Fine-Tune OpenAI models using W&B.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Fine-Tuning

With Weights & Biases you can log your OpenAI GPT-3.5 or GPT-4 model's fine-tuning metrics and configuration to Weights & Biases to analyse and understand the performance of your newly fine-tuned models and share the results with your colleagues. You can check out the models that can be fine-tuned [here](https://platform.openai.com/docs/guides/fine-tuning/what-models-can-be-fine-tuned).

:::info
The Weights and Biases fine-tuning integration works with `openai >= 1.0`. Please install the latest version of `openai` by doing `pip install -U openai`.
:::

## Sync your OpenAI Fine-Tuning Results in 2 Lines

If you use OpenAI's API to [fine-tune OpenAI models](https://platform.openai.com/docs/guides/fine-tuning/), you can now use the W&B integration to track experiments, models, and datasets in your central dashboard.

```python
from wandb.integration.openai import WandbLogger

## Finetuning logic

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

<!-- ![](/images/integrations/open_ai_api.png) -->
![](/images/integrations/open_ai_auto_scan.png)

### Check out interactive examples

* [Demo Colab](http://wandb.me/openai-colab)
* [Report - OpenAI Fine-Tuning Exploration and Tips](http://wandb.me/openai-report)

### Sync your fine-tunes with few lines of code

Make sure you are using latest version of openai and wandb.

```shell-session
pip install --upgrade openai wandb
```

Then sync your results from your script 


```python
from wandb.integration.openai import WandbLogger

# one line command
WandbLogger.sync()

# passing optional parameters
WandbLogger.sync(
    fine_tune_job_id=None,
    num_fine_tunes=None,
    project="OpenAI-Fine-Tune",
    entity=None,
    overwrite=False,
    **kwargs_wandb_init
)
```

### Reference

| Argument                 | Description                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| fine_tune_job_id         | This is the OpenAI Fine-Tune ID which you get when you create your fine-tune job using `client.fine_tuning.jobs.create`. If this argument is None (default), all the OpenAI fine-tune jobs that haven't already been synced will be synced to W&B.                                                                                        |
| openai_client            | Pass an initialized OpenAI client to `sync`. If no client is provided, one is initialized by the logger itself. By default it is None.                |
| num_fine_tunes           | If no ID is provided, then all the unsynced fine-tunes will be logged to W&B. This argument allows you to select the number of recent fine-tunes to sync. If num_fine_tunes is 5, it selects the 5 most recent fine-tunes.                                                  |
| project                  | Weights and Biases project name where your fine-tune metrics, models, data, etc. will be logged. By default, the project name is "OpenAI-Fine-Tune." |
| entity                   | Weights & Biases Username or team name where you're sending runs. By default, your default entity is used, which is usually your username. |
| overwrite                | Forces logging and overwrite existing wandb run of the same fine-tune job. By default this is False.                                                |
| wait_for_job_success     | Once an OpenAI fine-tuning job is started it usually takes a bit of time. To ensure that your metrics are logged to W&B as soon as the fine-tune job is finished, this setting will check every 60 seconds for the status of the fine-tune job to change to "succeeded". Once the fine-tune job is detected as being successful, the metrics will be synced automatically to W&B. Set to True by default.                                                    |
| \*\*kwargs\_wandb\_init  | Aany additional argument passed directly to [`wandb.init()`](../../../ref/python/init.md)                    |

## Dataset Versioning and Visualization

### Versioning

The training and validation data that you upload to OpenAI for fine-tuning are automatically logged as W&B Artifacts for easier version control. Below is an view of the training file in Artifacts. Here you can see the W&B run that logged this file, when it was logged, what version of the dataset this is, the metadata, and DAG lineage from the training data to the trained model.

![](/images/integrations/openai_data_artifacts.png)

### Visualization

The datasets are also visualized as W&B Tables which allows you to explore, search and interact with the dataset. Check out the training samples visualized using W&B Tables below.

![](/images/integrations/openai_data_visualization.png)


## The fine-tuned model and model versioning






## Frequently Asked Questions

### How do I share my fine-tune resutls with my team in W&B?

Sync all your runs to your team account with:

```shell-session
openai wandb sync --entity MY_TEAM_ACCOUNT
```

### How can I organize my runs?

Your W&B runs are automatically organized and can be filtered/sorted based on any configuration parameter such as job type, base model, learning rate, training filename and any other hyper-parameter.

In addition, you can rename your runs, add notes or create tags to group them.

Once you’re satisfied, you can save your workspace and use it to create report, importing data from your runs and saved artifacts (training/validation files).

### How can I access my fine-tune details?

Fine-tune details are logged to W&B as artifacts and can be accessed with:

```python
import wandb

ft_artifact = wandb.run.use_artifact('USERNAME/PROJECT/job_details:VERSION')
```

where `VERSION` is either:

* a version number such as `v2`
* the fine-tune id such as `ft-xxxxxxxxx`
* an alias added automatically such as `latest` or manually

You can then access fine-tune details through `artifact_job.metadata`. For example, the fine-tuned model can be retrieved with `artifact_job.metadata[`"`fine_tuned_model"]`.

### What if a fine-tune was not synced successfully?

You can always call again `openai wandb sync` and we will re-sync any run that was not synced successfully.

If needed, you can call `openai wandb sync --id fine_tune_id --force` to force re-syncing a specific fine-tune.

### Can I track my datasets with W&B?

Yes, you can integrate your entire pipeline to W&B through Artifacts, including creating your dataset, splitting it, training your models and evaluating them!

This will allow complete traceability of your models.

![](/images/integrations/open_ai_faq_can_track.png)

## Resources

* [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning/) is very thorough and contains many useful tips
* [Demo Colab](http://wandb.me/openai-colab)
* [Report - OpenAI Fine-Tuning Exploration & Tips](http://wandb.me/openai-report)
