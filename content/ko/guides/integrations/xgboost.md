---
description: Track your trees with W&B.
menu:
  default:
    identifier: ko-guides-integrations-xgboost
    parent: integrations
title: XGBoost
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

The `wandb` library has a `WandbCallback` callback for logging metrics, configs and saved boosters from training with XGBoost. Here you can see a [live W&B Dashboard](https://wandb.ai/morg/credit_scorecard) with outputs from the XGBoost `WandbCallback`.

{{< img src="/images/integrations/xgb_dashboard.png" alt="W&B Dashboard using XGBoost" >}}

## Get started

Logging XGBoost metrics, configs and booster models to W&B is as easy as passing the `WandbCallback` to XGBoost:

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# Start a wandb run
with wandb.init() as run:
  # Pass WandbCallback to the model
  bst = XGBClassifier()
  bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])
```

You can open [this notebook](https://wandb.me/xgboost) for a comprehensive look at logging with XGBoost and W&B

## `WandbCallback` reference

### Functionality
Passing `WandbCallback` to a XGBoost model will:
- log the booster model configuration to W&B
- log evaluation metrics collected by XGBoost, such as rmse, accuracy etc to W&B
- log training metrics collected by XGBoost (if you provide data to eval_set)
- log the best score and the best iteration
- save and upload your trained model to W&B Artifacts (when `log_model = True`)
- log feature importance plot when `log_feature_importance=True` (default).
- Capture the best eval metric in `wandb.Run.summary` when `define_metric=True` (default).

### Arguments
- `log_model`: (boolean) if True save and upload the model to W&B Artifacts

- `log_feature_importance`: (boolean) if True log a feature importance bar plot

- `importance_type`: (str) one of `{weight, gain, cover, total_gain, total_cover}` for tree model. weight for linear model.

- `define_metric`: (boolean) if True (default) capture model performance at the best step, instead of the last step, of training in your `run.summary`.


You can review the [source code for WandbCallback](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py).

For additional examples, check out the [repository of examples on GitHub](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms).

## Tune your hyperparameters with Sweeps

Attaining the maximum performance out of models requires tuning hyperparameters, like tree depth and learning rate. W&B [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) is a powerful toolkit for configuring, orchestrating, and analyzing large hyperparameter testing experiments.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

You can also try this [XGBoost & Sweeps Python script](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py).

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="XGBoost performance comparison" >}}