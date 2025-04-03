---
title: DeepChecks
description: DeepChecks と W&B の統合方法。
menu:
  default:
    identifier: ja-guides-integrations-deepchecks
    parent: integrations
weight: 60
---

{{< cta-button colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb" >}}

DeepChecks は、機械学習 モデル と データの検証を支援します。たとえば、データの整合性の検証、分布の検査、データ分割の検証、モデル の評価、異なる モデル 間の比較などを、最小限の労力で行うことができます。

[DeepChecks と wandb の インテグレーション についてもっと読む ->](https://docs.deepchecks.com/stable/general/usage/exporting_results/auto_examples/plot_exports_output_to_wandb.html)

## はじめに

DeepChecks を Weights & Biases とともに使用するには、まず Weights & Biases アカウント にサインアップする 必要があります [こちら](https://wandb.ai/site)。DeepChecks の Weights & Biases の インテグレーション を使用すると、次のよう にすぐに開始できます。

```python
import wandb

wandb.login()

# deepchecks から チェック をインポートします
from deepchecks.checks import ModelErrorAnalysis

# チェック を実行します
result = ModelErrorAnalysis()

# その 結果 を wandb にプッシュします
result.to_wandb()
```

DeepChecks テストスイート 全体 を Weights & Biases に ログ することもできます

```python
import wandb

wandb.login()

# deepchecks から full_suite テスト をインポートします
from deepchecks.suites import full_suite

# DeepChecks テストスイート を作成して実行します
suite_result = full_suite().run(...)

# thes の 結果 を wandb にプッシュします
# ここでは、必要な wandb.init の config と 引数 を渡すことができます
suite_result.to_wandb(project="my-suite-project", config={"suite-name": "full-suite"})
```

## 例

``[**この レポート**](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5) は、DeepChecks と Weights & Biases を使用する威力を示しています

{{< img src="/images/integrations/deepchecks_example.png" alt="" >}}

この Weights & Biases の インテグレーション に関する質問や問題がありますか？ [DeepChecks github repository](https://github.com/deepchecks/deepchecks) で issue をオープンしてください。私たちがキャッチして回答します :)
