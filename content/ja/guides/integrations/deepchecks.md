---
title: DeepChecks
description: W&B と DeepChecks を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-deepchecks
    parent: integrations
weight: 60
---

{{< cta-button colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb" >}}
DeepChecks は、機械学習 の モデル と データ の検証を支援します。データの整合性の確認、分布の点検、データ分割の検証、モデルの評価や異なるモデル間の比較までを、最小限の手間で行えます。
[DeepChecks と W&B インテグレーションの詳細はこちら ->](https://docs.deepchecks.com/stable/general/usage/exporting_results/auto_examples/plot_exports_output_to_wandb.html)
## はじめに
DeepChecks を W&B と 一緒に 使うには、まず [W&B アカウント](https://wandb.ai/site) に登録してください。DeepChecks の W&B インテグレーションを使えば、次のようにすぐに始められます:
```python
import wandb

wandb.login()

# deepchecks からチェックをインポート
from deepchecks.checks import ModelErrorAnalysis

# チェックを実行
result = ModelErrorAnalysis()

# 結果を W&B に送信
result.to_wandb()
```
DeepChecks のテストスイート全体を W&B にログとして記録することもできます。
```python
import wandb

wandb.login()

# deepchecks から full_suite テストをインポート
from deepchecks.suites import full_suite

# DeepChecks のテストスイートを作成して実行
suite_result = full_suite().run(...)

# これらの結果を W&B に送信
# ここでは必要な wandb.init の 設定や引数を渡せます
suite_result.to_wandb(project="my-suite-project", config={"suite-name": "full-suite"})
```
## 例
[この Report](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5) では、DeepChecks と W&B を使う強力なワークフローを紹介しています。
{{< img src="/images/integrations/deepchecks_example.png" alt="DeepChecks のデータ検証結果" >}}
この W&B インテグレーションについて質問や問題はありますか？[DeepChecks の GitHub リポジトリ](https://github.com/deepchecks/deepchecks) に Issue を作成してください。確認のうえ、回答をお届けします。