---
title: DeepChecks
description: W&B を DeepChecks と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-deepchecks
    parent: integrations
weight: 60
---

{{< cta-button colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb" >}}

DeepChecks は、データの整合性の確認、分布の検査、データ分割の検証、モデルの評価や異なるモデル間の比較など、機械学習のモデルとデータを検証するのに役立ちます。すべて最小限の労力で実行できます。

[DeepChecks と wandb インテグレーションについてもっと読む ->](https://docs.deepchecks.com/stable/general/usage/exporting_results/auto_examples/plot_exports_output_to_wandb.html)

## 始めるにあたって

DeepChecks を Weights & Biases と共に使用するには、まず [こちら](https://wandb.ai/site) から Weights & Biases アカウントに登録する必要があります。 DeepChecks の Weights & Biases インテグレーションを使用すると、次のようにすぐに始めることができます。

```python
import wandb

wandb.login()

# deepchecks からチェックをインポートする
from deepchecks.checks import ModelErrorAnalysis

# チェックを実行する
result = ModelErrorAnalysis()

# その結果を wandb にプッシュする
result.to_wandb()
```

DeepChecks のテストスイート全体を Weights & Biases にログすることもできます

```python
import wandb

wandb.login()

# deepchecks から full_suite テストをインポートする
from deepchecks.suites import full_suite

# DeepChecks テストスイートを作成および実行する
suite_result = full_suite().run(...)

# その結果を wandb にプッシュする
# ここで必要な wandb.init の構成と引数を渡すことができます
suite_result.to_wandb(project="my-suite-project", config={"suite-name": "full-suite"})
```

## 例

``[**このレポート**](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5) は、DeepChecks と Weights & Biases を使用する力を示しています。

{{< img src="/images/integrations/deepchecks_example.png" alt="" >}}

この Weights & Biases インテグレーションについて質問や問題がありますか？ [DeepChecks github リポジトリ](https://github.com/deepchecks/deepchecks) でイシューを開いていただければ、対応し、回答をお届けします :)