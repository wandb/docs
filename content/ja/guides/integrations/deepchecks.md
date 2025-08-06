---
title: DeepChecks（ディープチェック）
description: W&B を DeepChecks と統合する方法
menu:
  default:
    identifier: deepchecks
    parent: integrations
weight: 60
---

{{< cta-button colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb" >}}

DeepChecks を使うと、機械学習モデルやデータの検証が簡単に行えます。たとえば、データの完全性チェック、分布の確認、データ分割の検証、モデルの評価、複数モデル間の比較などが、最小限の手間で実現できます。

[DeepChecks と wandb のインテグレーションについて詳しくはこちら ->](https://docs.deepchecks.com/stable/general/usage/exporting_results/auto_examples/plot_exports_output_to_wandb.html)

## はじめに

DeepChecks を W&B で利用するには、まず [W&B アカウント](https://wandb.ai/site) に登録してください。DeepChecks の W&B インテグレーションを使えば、すぐに始められます:

```python
import wandb

wandb.login()

# deepchecks からチェックをインポート
from deepchecks.checks import ModelErrorAnalysis

# チェックを実行
result = ModelErrorAnalysis()

# 結果を wandb へ送信
result.to_wandb()
```

DeepChecks のテストスイート全体を W&B へログすることも可能です。

```python
import wandb

wandb.login()

# deepchecks から full_suite テストをインポート
from deepchecks.suites import full_suite

# DeepChecks テストスイートを作成・実行
suite_result = full_suite().run(...)

# 結果を wandb へ送信
# ここで wandb.init の設定や引数も自由に渡せます
suite_result.to_wandb(project="my-suite-project", config={"suite-name": "full-suite"})
```

## 例

[この Report](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5) では、DeepChecks と W&B の組み合わせによる強力な機能を紹介しています。

{{< img src="/images/integrations/deepchecks_example.png" alt="Deepchecks データバリデーションの結果" >}}

W&B インテグレーションについて質問や問題があれば、[DeepChecks の GitHub リポジトリ](https://github.com/deepchecks/deepchecks) にイシューを立ててください。対応いたします。