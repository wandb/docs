---
description: W&BをDeepChecksと統合する方法
slug: /guides/integrations/deepchecks
displayed_sidebar: default
---


# DeepChecks

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export\_outputs\_to\_wandb.ipynb)

DeepChecksは、データの整合性を検証したり、データの分布を調べたり、データ分割を検証したり、モデルを評価したり、異なるモデルを比較したりと、機械学習モデルやデータの検証を最小限の労力で行うことができます。

[DeepChecksとwandbインテグレーションについてさらに読む ->](https://docs.deepchecks.com/en/stable/examples/guides/export\_outputs\_to\_wandb.html)

## 始め方

DeepChecksをWeights & Biasesと一緒に使用するには、まず [ここ](https://wandb.ai/site) からWeights & Biasesのアカウントにサインアップする必要があります。DeepChecksのWeights & Biasesインテグレーションを使用すると、以下のようにすぐに始めることができます：

```python
import wandb
wandb.login()

# deepchecks からチェックをインポート
from deepchecks.checks import ModelErrorAnalysis

# チェックを実行
result = ModelErrorAnalysis()...

# その結果を wandb にプッシュ
result.to_wandb()
```

DeepChecksの全テストスイートをWeights & Biasesにログすることもできます

```python
import wandb
wandb.login()

# deepchecks から full_suite テストをインポート
from deepchecks.suites import full_suite

# DeepChecksテストスイートを作成して実行
suite_result = full_suite().run(...)

# 結果を wandb にプッシュ
# ここで必要な wandb.init 設定や引数を渡すことができます
suite_result.to_wandb(
    project='my-suite-project', 
    config={'suite-name': 'full-suite'}
)
```

## 例

``[**このレポート**](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5) では、DeepChecksとWeights & Biasesの使用の威力を示しています

![](/images/integrations/deepchecks_example.png)

Weights & Biases のインテグレーションに関する質問や問題がある場合は、 [DeepChecksのgithubリポジトリ](https://github.com/deepchecks/deepchecks) に問題を提出してください。私たちが確認し、回答いたします :)