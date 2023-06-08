---
slug: /guides/integrations/deepchecks
description: How to integrate W&B with DeepChecks.
displayed_sidebar: ja
---

# DeepChecks

[![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export\_outputs\_to\_wandb.ipynb)

DeepChecksは、データの整合性の検証、分布の検査、データ分割の検証、モデルの評価、異なるモデル間の比較など、機械学習モデルとデータの検証を簡単に行うことができます。

[DeepChecksとwandb統合についてさらに読む ->](https://docs.deepchecks.com/en/stable/examples/guides/export\_outputs\_to\_wandb.html)

## はじめに

Weights & BiasesとDeepChecksを一緒に使うには、まずこちらからWeights & Biasesのアカウントに登録してください。[こちら](https://wandb.ai/site)。Weights & BiasesとDeepChecksを統合することで、次のように簡単に始めることができます。

```python
import wandb
wandb.login()

# deepchecksからチェックをインポートする
from deepchecks.checks import ModelErrorAnalysis

# チェックを実行する
result = ModelErrorAnalysis()...

# その結果をwandbにプッシュする
result.to_wandb()
```
Weights & Biases に DeepChecks のテストスイート全体をログすることもできます

```python
import wandb

wandb.login()

# deepchecks から full_suite テストをインポート
from deepchecks.suites import full_suite

# DeepChecks のテストスイートを作成して実行
suite_result = full_suite().run(...)

# これらの結果を wandb にプッシュ
# ここで必要な wandb.init の設定と引数を渡すことができます
suite_result.to_wandb(
    project='my-suite-project',
    config={'suite-name': 'full-suite'}
)
```

## 例

[**このレポート**](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5) は、DeepChecks と Weights & Biases を使った力を示しています。

![](/images/integrations/deepchecks_example.png)

この Weights & Biases の統合についての質問や問題がありますか？[DeepChecks の github リポジトリ](https://github.com/deepchecks/deepchecks)で issue を開いてください。それを確認し、解答をお届けします :)