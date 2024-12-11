---
description: Track your trees with W&B.
title: LightGBM
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb'/>

The `wandb` library includes a special callback for [LightGBM](https://lightgbm.readthedocs.io/en/latest/). It's also easy to use the generic logging features of Weights & Biases to track large experiments, like hyperparameter sweeps.

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# Log metrics to W&B
gbm = lgb.train(..., callbacks=[wandb_callback()])

# Log feature importance plot and upload model checkpoint to W&B
log_summary(gbm, save_model_checkpoint=True)
```

:::info
Looking for working code examples? Check out [our repository of examples on GitHub](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms).
:::

## Tuning your hyperparameters with Sweeps

Attaining the maximum performance out of models requires tuning hyperparameters, like tree depth and learning rate. Weights & Biases includes [Sweeps](../sweeps/), a powerful toolkit for configuring, orchestrating, and analyzing large hyperparameter testing experiments.

:::info
To learn more about these tools and see an example of how to use Sweeps with XGBoost, check out this interactive Colab notebook.

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb'/>
:::

![tl;dr: trees outperform linear learners on this classification dataset.](/images/integrations/lightgbm_sweeps.png)
