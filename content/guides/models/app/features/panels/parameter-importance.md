---
description: Visualize the relationships between your model's hyperparameters and
  output metrics
menu:
  default:
    identifier: parameter-importance
    parent: panels
title: Parameter importance
weight: 60
---

Discover which of your hyperparameters were the best predictors of, and highly correlated to desirable values of your metrics.


{{< img src="/images/general/parameter-importance-1.png" alt="" >}}

**Correlation** is the linear correlation between the hyperparameter and the chosen metric (in this case val_loss). So a high correlation means that when the hyperparameter has a higher value, the metric also has higher values and vice versa. Correlation is a great metric to look at but it can’t capture second order interactions between inputs and it can get messy to compare inputs with wildly different ranges.

Therefore W&B also calculates an **importance** metric. W&B trains a random forest with the hyperparameters as inputs and the metric as the target output and report the feature importance values for the random forest.

The idea for this technique was inspired by a conversation with [Jeremy Howard](https://twitter.com/jeremyphoward) who has pioneered the use of random forest feature importances to explore hyperparameter spaces at [Fast.ai](http://fast.ai). W&B highly recommends you check out this [lecture](http://course18.fast.ai/lessonsml1/lesson4.html) (and these [notes](https://forums.fast.ai/t/wiki-lesson-thread-lesson-4/7540)) to learn more about the motivation behind this analysis.

Hyperparameter importance panel untangles the complicated interactions between highly correlated hyperparameters. In doing so, it helps you fine tune your hyperparameter searches by showing you which of your hyperparameters matter the most in terms of predicting model performance.

## Creating a hyperparameter importance panel

1. Navigate to your W&B project.
2. Select **Add panels** buton.
3. Expand the **CHARTS** dropdown, choose **Parallel coordinates** from the dropdown.


{{% alert %}}
If an empty panel appears, make sure that your runs are ungrouped
{{% /alert %}}


{{< img src="/images/app_ui/hyperparameter_importance_panel.gif" alt="Using automatic parameter visualization" >}}

With the parameter manager, we can manually set the visible and hidden parameters.

{{< img src="/images/app_ui/hyperparameter_importance_panel_manual.gif" alt="Manually setting the visible and hidden fields" >}}

## Interpreting a hyperparameter importance panel

{{< img src="/images/general/parameter-importance-4.png" alt="" >}}

This panel shows you all the parameters passed to the [wandb.config](/guides/track/config/) object in your training script. Next, it shows the feature importances and correlations of these config parameters with respect to the model metric you select (`val_loss` in this case).

### Importance

The importance column shows you the degree to which each hyperparameter was useful in predicting the chosen metric. Imagine a scenario were you start tuning a plethora of hyperparameters and using this plot to hone in on which ones merit further exploration. The subsequent sweeps can then be limited to the most important hyperparameters, thereby finding a better model faster and cheaper.

{{% alert %}}
W&B calculate importances using a tree based model rather than a linear model as the former are more tolerant of both categorical data and data that’s not normalized.
{{% /alert %}}

In the preceding image, you can see that `epochs, learning_rate, batch_size` and `weight_decay` were fairly important.

### Correlations

Correlations capture linear relationships between individual hyperparameters and metric values. They answer the question of whether there a significant relationship between using a hyperparameter, such as the SGD optimizer, and the `val_loss` (the answer in this case is yes). Correlation values range from -1 to 1, where positive values represent positive linear correlation, negative values represent negative linear correlation and a value of 0 represents no correlation. Generally a value greater than 0.7 in either direction represents strong correlation.

You might use this graph to further explore the values that are have a higher correlation to our metric (in this case you might pick stochastic gradient descent or adam over rmsprop or nadam) or train for more epochs.


{{% alert %}}
* correlations show evidence of association, not necessarily causation.
* correlations are sensitive to outliers, which might turn a strong relationship to a moderate one, specially if the sample size of hyperparameters tried is small.
* and finally, correlations only capture linear relationships between hyperparameters and metrics. If there is a strong polynomial relationship, it won’t be captured by correlations.
{{% /alert %}}

The disparities between importance and correlations result from the fact that importance accounts for interactions between hyperparameters, whereas correlation only measures the affects of individual hyperparameters on metric values. Secondly, correlations capture only the linear relationships, whereas importances can capture more complex ones.

As you can see both importance and correlations are powerful tools for understanding how your hyperparameters influence model performance.