---
slug: /guides/integrations/openai
description: How to Fine-Tune OpenAI models using W&B.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Fine-Tuning

Log your OpenAI model's fine-tuning metrics and configuration to Weights & Biases to analyse and understand the performance of your newly fine-tuned models and share the results with your colleagues.

## Sync your OpenAI Fine-Tuning Results in 1 Line

If you use OpenAI's API to [fine-tune OpenAI models](https://platform.openai.com/docs/guides/fine-tuning/), you can now use the W&B integration to track experiments, models, and datasets in your central dashboard.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Python', value: 'python_sdk'},
  ]}>
  <TabItem value="cli">

```shell-session
openai wandb sync
```

  </TabItem>
  <TabItem value="python_sdk">

```python
from openai.wandb_logger import WandbLogger

WandbLogger.sync(project="OpenAI-Fine-Tune")
```
  </TabItem>
</Tabs>

<!-- ![](/images/integrations/open_ai_api.png) -->
![](/images/integrations/open_ai_auto_scan.png)

### :sparkles: Check out interactive examples

* [Demo Colab](http://wandb.me/openai-colab)
* [Report - OpenAI Fine-Tuning Exploration and Tips](http://wandb.me/openai-report)

### :tada: Sync your fine-tunes with one line!

Make sure you are using latest version of openai and wandb.

```shell-session
$ pip install --upgrade openai wandb
```

Then sync your results from the command line or from your script.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Python', value: 'python_sdk'},
  ]}>
  <TabItem value="cli">

```shell-session
$ # one line command
$ openai wandb sync

$ # passing optional parameters
$ openai wandb sync --help
```
  </TabItem>
  <TabItem value="python_sdk">

```python
from openai.wandb_logger import WandbLogger

# one line command
WandbLogger.sync()

# passing optional parameters
WandbLogger.sync(
    id=None,
    n_fine_tunes=None,
    project="OpenAI-Fine-Tune",
    entity=None,
    force=False,
    **kwargs_wandb_init
)
```
  </TabItem>
</Tabs>

When you sync your results, wandb checks OpenAI for newly completed fine-tunes and automatically adds them to your dashboard.

In addition your training and validation files are logged and versioned, as well as details of your fine-tune results. This let you interactively explore your training and validation data.

![](/images/integrations/open_ai_validation_files.png)

### :gear: Optional arguments

| Argument                 | Description                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| -i ID, --id ID           | The id of the fine-tune (optional)                                                                                        |
| -n N, --n\_fine\_tunes N | Number of most recent fine-tunes to log when an id is not provided. By default, every fine-tune is synced.                |
| --project PROJECT        | Name of the project where you're sending runs. By default, it is "GPT-3".                                                 |
| --entity ENTITY          | Username or team name where you're sending runs. By default, your default entity is used, which is usually your username. |
| --force                  | Forces logging and overwrite existing wandb run of the same fine-tune.                                                    |
| --legacy                  | Log results from the legacy OpenAI GPT-3 fine-tune api.                                                    |
| \*\*kwargs\_wandb\_init  | In python, any additional argument is directly passed to [`wandb.init()`](../../../ref/python/init.md)                    |

### 🔍 Inspect sample predictions

Use [Tables](../../tables/intro.md) to better visualize sample predictions and compare models.

![](/images/integrations/open_ai_inspect_sample.png)

Create a new run:

```python
run = wandb.init(project="OpenAI-Fine-Tune", job_type="eval")
```

Retrieve a model id for inference.

You can use automatically logged artifacts to retrieve your latest model:

```python
ft_artifact = run.use_artifact("ENTITY/PROJECT/fine_tune_details:latest")
fine_tuned_model = ft_artifact.metadata["fine_tuned_model"]
```

You can also retrieve your validation file:

```python
artifact_valid = run.use_artifact("ENTITY/PROJECT/FILENAME:latest")
valid_file = artifact_valid.get_path("FILENAME").download()
```

Perform some inferences using OpenAI API:

```python
# perform inference and record results
my_prompts = ["PROMPT_1", "PROMPT_2"]
results = []
for prompt in my_prompts:
    res = openai.ChatCompletion.create(model=fine_tuned_model,
                                   prompt=prompt,
                                   ...)
    results.append(res["choices"][0]["text"])
```

Log your results with a Table:

```python
table = wandb.Table(columns=['prompt', 'completion'],
                    data=list(zip(my_prompts, results)))
```

## :question:Frequently Asked Questions

### How do I share my fine-tune resutls with my team in W&B?

Sync all your runs to your team account with:

```shell-session
$ openai wandb sync --entity MY_TEAM_ACCOUNT
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

## :books: Resources

* [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning/) is very thorough and contains many useful tips
* [Demo Colab](http://wandb.me/openai-colab)
* [Report - OpenAI Fine-Tuning Exploration & Tips](http://wandb.me/openai-report)
