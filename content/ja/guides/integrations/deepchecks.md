---
title: DeepChecks
description: DeepChecks と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-deepchecks
    parent: integrations
weight: 60
---

{{< cta-button colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb" >}}

DeepChecks は、機械学習 モデルとデータを検証するのに役立ちます。たとえば、データの整合性の検証、分布の検査、データ分割の検証、モデルの評価、異なるモデル間の比較などを、最小限の労力で行うことができます。

[DeepChecks と wandb の インテグレーション の詳細はこちら ->](https://docs.deepchecks.com/stable/general/usage/exporting_results/auto_examples/plot_exports_output_to_wandb.html)

## 始め方

Weights & Biases で DeepChecks を使用するには、まず Weights & Biases のアカウントにサインアップする 必要があります [こちら](https://wandb.ai/site)。DeepChecks の Weights & Biases インテグレーション を使用すると、次の ようにすぐに始めることができます。

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

DeepChecks テストスイート全体を Weights & Biases に ログ することもできます。

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

## 例

``[**This Report**](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5) は、DeepChecks と Weights & Biases を使用することの威力を示しています。

{{< img src="/images/integrations/deepchecks_example.png" alt="" >}}

この Weights & Biases の インテグレーション についての質問や問題がありますか？ [DeepChecks github repository](https://github.com/deepchecks/deepchecks) で issue を オープン してください。こちらで確認して回答いたします :)
