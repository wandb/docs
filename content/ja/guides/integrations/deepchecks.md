---
title: DeepChecks
description: W&B を DeepChecks と連携する方法
menu:
  default:
    identifier: ja-guides-integrations-deepchecks
    parent: integrations
weight: 60
---

{{< cta-button colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb" >}}

DeepChecks は、機械学習モデルやデータの検証を支援し、データの整合性チェックや分布の確認、データ分割のバリデーション、モデルの評価や異なるモデル間の比較などを、最小限の手間で実施できます。

[DeepChecksと wandb インテグレーションの詳細はこちら →](https://docs.deepchecks.com/stable/general/usage/exporting_results/auto_examples/plot_exports_output_to_wandb.html)

## はじめに

DeepChecks を W&B で利用するには、まず[W&B アカウント](https://wandb.ai/site)に登録する必要があります。DeepChecks の W&B インテグレーションを使うと、以下のように素早く利用開始できます。

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

また、DeepChecks のテストスイート全体を W&B にログすることもできます。

```python
import wandb

wandb.login()

# deepchecks から full_suite テストをインポート
from deepchecks.suites import full_suite

# DeepChecks テストスイートを作成・実行
suite_result = full_suite().run(...)

# 結果を wandb へ送信
# ここでは任意の wandb.init 設定や引数を渡すことができます
suite_result.to_wandb(project="my-suite-project", config={"suite-name": "full-suite"})
```

## 例

[この Reports](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5) は、DeepChecks と W&B を組み合わせたパワフルな使い方を紹介しています。

{{< img src="/images/integrations/deepchecks_example.png" alt="Deepchecks データバリデーション結果" >}}

W&B インテグレーションについて質問や問題がある場合は、[DeepChecks github リポジトリ](https://github.com/deepchecks/deepchecks) に issue を投稿してください。私たちが確認して回答いたします。