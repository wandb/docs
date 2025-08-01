---
description: How to Fine-Tune OpenAI models using W&B.
menu:
  default:
    identifier: openai-fine-tuning
    parent: integrations
title: OpenAI Fine-Tuning
weight: 250
---
{{< cta-button colabLink="https://wandb.me/openai-colab" >}}

Log your OpenAI GPT-3.5 or GPT-4 model's fine-tuning metrics and configuration to W&B. Utilize the W&B ecosystem to track your fine-tuning experiments, models, and datasets and share your results with your colleagues.

{{% alert %}}
See the [OpenAI documentation](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned) for a list of models that you can fine tune.
{{% /alert %}}

See the [Weights and Biases Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration) section in the OpenAI documentation for supplemental information on how to integrate W&B with OpenAI for fine-tuning.


## Install or update OpenAI Python API

The W&B OpenAI fine-tuning integration works with OpenAI version 1.0 and above. See the PyPI documentation for the latest version of the [OpenAI Python API](https://pypi.org/project/openai/) library.


To install OpenAI Python API, run:
```python
pip install openai
```

If you already have OpenAI Python API installed, you can update it with:
```python
pip install -U openai
```


## Sync your OpenAI fine-tuning results

Integrate W&B with OpenAI's fine-tuning API to log your fine-tuning metrics and configuration to W&B. To do this, use the `WandbLogger` class from the `wandb.integration.openai.fine_tuning` module.


```python
from wandb.integration.openai.fine_tuning import WandbLogger

# Finetuning logic

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="OpenAI auto-scan feature" >}}

### Sync your fine-tunes

Sync your results from your script 


```python
from wandb.integration.openai.fine_tuning import WandbLogger

# one line command
WandbLogger.sync()

# passing optional parameters
WandbLogger.sync(
    fine_tune_job_id=None,
    num_fine_tunes=None,
    project="OpenAI-Fine-Tune",
    entity=None,
    overwrite=False,
    model_artifact_name="model-metadata",
    model_artifact_type="model",
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
| entity                   | W&B Username or team name where you're sending runs. By default, your default entity is used, which is usually your username. |
| overwrite                | Forces logging and overwrite existing wandb run of the same fine-tune job. By default this is False.                                                |
| wait_for_job_success     | Once an OpenAI fine-tuning job is started it usually takes a bit of time. To ensure that your metrics are logged to W&B as soon as the fine-tune job is finished, this setting will check every 60 seconds for the status of the fine-tune job to change to `succeeded`. Once the fine-tune job is detected as being successful, the metrics will be synced automatically to W&B. Set to True by default.                                                    |
| model_artifact_name      | The name of the model artifact that is logged. Defaults to `"model-metadata"`.                    |
| model_artifact_type      | The type of the model artifact that is logged. Defaults to `"model"`.                    |
| \*\*kwargs_wandb_init  | Aany additional argument passed directly to [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}})                    |

## Dataset Versioning and Visualization

### Versioning

The training and validation data that you upload to OpenAI for fine-tuning are automatically logged as W&B Artifacts for easier version control. Below is an view of the training file in Artifacts. Here you can see the W&B run that logged this file, when it was logged, what version of the dataset this is, the metadata, and DAG lineage from the training data to the trained model.

{{< img src="/images/integrations/openai_data_artifacts.png" alt="W&B Artifacts with training datasets" >}}

### Visualization

The datasets are visualized as W&B Tables, which allows you to explore, search, and interact with the dataset. Check out the training samples visualized using W&B Tables below.

{{< img src="/images/integrations/openai_data_visualization.png" alt="OpenAI data" >}}


## The fine-tuned model and model versioning

OpenAI gives you an id of the fine-tuned model. Since we don't have access to the model weights, the `WandbLogger` creates a `model_metadata.json` file with all the details (hyperparameters, data file ids, etc.) of the model along with the `fine_tuned_model`` id and is logged as a W&B Artifact. 

This model (metadata) artifact can further be linked to a model in the [W&B Registry]({{< relref "/guides/core/registry/" >}}).

{{< img src="/images/integrations/openai_model_metadata.png" alt="OpenAI model metadata" >}}


## Frequently Asked Questions

### How do I share my fine-tune results with my team in W&B?

Log your fine-tune jobs to your team account with:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### How can I organize my runs?

Your W&B runs are automatically organized and can be filtered/sorted based on any configuration parameter such as job type, base model, learning rate, training filename and any other hyper-parameter.

In addition, you can rename your runs, add notes or create tags to group them.

Once you’re satisfied, you can save your workspace and use it to create report, importing data from your runs and saved artifacts (training/validation files).

### How can I access my fine-tuned model?

Fine-tuned model ID is logged to W&B as artifacts (`model_metadata.json`) as well config.

```python
import wandb
    
with wandb.init(project="OpenAI-Fine-Tune", entity="YOUR_TEAM_NAME") as run:
    ft_artifact = run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
    artifact_dir = ft_artifact.download()
```

where `VERSION` is either:

* a version number such as `v2`
* the fine-tune id such as `ft-xxxxxxxxx`
* an alias added automatically such as `latest` or manually

You can then access `fine_tuned_model` id by reading the downloaded `model_metadata.json` file.

### What if a fine-tune was not synced successfully?

If a fine-tune was not logged to W&B successfully, you can use the `overwrite=True` and pass the fine-tune job id:

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### Can I track my datasets and models with W&B?

The training and validation data are logged automatically to W&B as artifacts. The metadata including the ID for the fine-tuned model is also logged as artifacts.

You can always control the pipeline using low level wandb APIs like `wandb.Artifact`, `wandb.Run.log`, etc. This will allow complete traceability of your data and models.

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="OpenAI tracking FAQ" >}}

## Resources

* [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning/) is very thorough and contains many useful tips
* [Demo Colab](https://wandb.me/openai-colab)
* [How to Fine-Tune Your OpenAI GPT-3.5 and GPT-4 Models with W&B](https://wandb.me/openai-report) report