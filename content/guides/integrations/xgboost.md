---
description: Track your trees with W&B.
menu:
  default:
    identifier: xgboost
    parent: integrations
title: XGBoost
weight: 460
---
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

The `wandb` library has a `WandbCallback` callback for logging metrics, configs and saved boosters from training with XGBoost. Here you can see a **[live Weights & Biases dashboard](https://wandb.ai/morg/credit_scorecard)** with outputs from the XGBoost `WandbCallback`.

{{< img src="/images/integrations/xgb_dashboard.png" alt="Weights & Biases dashboard using XGBoost" >}}

## Get Started

Logging XGBoost metrics, configs and booster models to Weights & Biases is as easy as passing the `WandbCallback` to XGBoost:

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# Start a wandb run
run = wandb.init()

# Pass WandbCallback to the model
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# Close your wandb run
run.finish()
```

You can open **[this notebook](https://wandb.me/xgboost)** for a comprehensive look at logging with XGBoost and Weights & Biases

## WandbCallback

### Functionality
Passing `WandbCallback` to a XGBoost model will:
- log the booster model configuration to Weights & Biases
- log evaluation metrics collected by XGBoost, such as rmse, accuracy etc to Weights & Biases
- log training metrics collected by XGBoost (if you provide data to eval_set)
- log the best score and the best iteration
- save and upload your trained model to to Weights & Biases Artifacts (when `log_model = True`)
- log feature importance plot when `log_feature_importance=True` (default).
- Capture the best eval metric in `wandb.summary` when `define_metric=True` (default).

### Arguments
`log_model`: (boolean) if True save and upload the model to Weights & Biases Artifacts

`log_feature_importance`: (boolean) if True log a feature importance bar plot

`importance_type`: (str) one of `{weight, gain, cover, total_gain, total_cover}` for tree model. weight for linear model.

`define_metric`: (boolean) if True (default) capture model performance at the best step, instead of the last step, of training in your `wandb.summary`.


You can find the source code for WandbCallback [here](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)

{{% alert %}}
Looking for more working code examples? Check out [our repository of examples on GitHub](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms).
{{% /alert %}}

## Tuning your hyperparameters with Sweeps

Attaining the maximum performance out of models requires tuning hyperparameters, like tree depth and learning rate. Weights & Biases includes [Sweeps](../sweeps/intro.md), a powerful toolkit for configuring, orchestrating, and analyzing large hyperparameter testing experiments.

{{% alert %}}
See the following Colab notebook to learn more about these tools and see an example of how to use Sweeps with XGBoost.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

You can also try this [XGBoost & Sweeps Python script](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)
{{% /alert %}}

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="tl;dr: trees outperform linear learners on this classification dataset." >}}