---
slug: /guides/integrations/cohere-fine-tunining
description: How to Fine-Tune Cohere models using W&B.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Cohere fine-tuning

With Weights & Biases you can log your Cohere model's fine-tuning metrics and configuration to analyze and understand the performance of your models and share the results with your colleagues. 

This [guide from Cohere](https://docs.cohere.com/page/convfinqa-finetuning-wandb) has a full example of how to kick off a fine-tuning run and you can find the [Cohere api docs here](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb)

## Log your Cohere fine-tuning results

To quickly add Cohere fine-tuning logging to your W&B workspace:

1. Create a `WandbConfig` with your W&B API key, W&B `entity` and `project` name. You can find your W&B API key at https://wandb.ai/authorize

2. Pass this config to the `FinetunedModel` object along with your model name, dataset and hyperparameters to kick off your fine-tuning run.


```python
from cohere.finetuning import WandbConfig, FinetunedModel

# create a config with your W&B details
wandb_ft_config = WandbConfig(
    api_key="<wandb_api_key>",
    entity="my-entity", # must be a valid enitity associated with the provided API key
    project="cohere-ft",
)

...  # set up your datasets and hyperparameters

# start a fine-tuning run on cohere
cmd_r_finetune = co.finetuning.create_finetuned_model(
  request=FinetunedModel(
    name="command-r-ft",
    settings=Settings(
      base_model=...
      dataset_id=...
      hyperparameters=...
      wandb=wandb_ft_config  # pass your W&B config here
    ),
  ),
)
```

You can then view your model's fine-tuning training and validation metrics and hyperparameters in the W&B project that you created.

![](/images/integrations/cohere_ft.png)


## Frequently Asked Questions

### How can I organize my runs?

Your W&B runs are automatically organized and can be filtered/sorted based on any configuration parameter such as job type, base model, learning rate and any other hyper-parameter.

In addition, you can rename your runs, add notes or create tags to group them.


## Resources

* **[Cohere Fine-tuning Example](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)** 