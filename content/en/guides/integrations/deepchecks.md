---
description: How to integrate W&B with DeepChecks.
menu:
  default:
    identifier: deepchecks
    parent: integrations
title: DeepChecks
weight: 60
---
{{< cta-button colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb" >}}

DeepChecks helps you validate your machine learning models and data, such as verifying your data’s integrity, inspecting its distributions, validating data splits, evaluating your model and comparing between different models, all with minimal effort.

[Read more about DeepChecks and the wandb integration ->](https://docs.deepchecks.com/stable/general/usage/exporting_results/auto_examples/plot_exports_output_to_wandb.html)

## Getting Started

To use DeepChecks with Weights & Biases you will first need to sign up for a [Weights & Biases account](https://wandb.ai/site). With the Weights & Biases integration in DeepChecks you can quickly get started like so:

```python
import wandb

wandb.login()

# import your check from deepchecks
from deepchecks.checks import ModelErrorAnalysis

# run your check
result = ModelErrorAnalysis()

# push that result to wandb
result.to_wandb()
```

You can also log an entire DeepChecks test suite to Weights & Biases

```python
import wandb

wandb.login()

# import your full_suite tests from deepchecks
from deepchecks.suites import full_suite

# create and run a DeepChecks test suite
suite_result = full_suite().run(...)

# push thes results to wandb
# here you can pass any wandb.init configs and arguments you need
suite_result.to_wandb(project="my-suite-project", config={"suite-name": "full-suite"})
```

## Example

[This Report](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5) shows off the power of using DeepChecks and Weights & Biases.

{{< img src="/images/integrations/deepchecks_example.png" alt="Deepchecks data validation results" >}}

Any questions or issues about this Weights & Biases integration? Open an issue in the [DeepChecks github repository](https://github.com/deepchecks/deepchecks) and we'll catch it and get you an answer :)