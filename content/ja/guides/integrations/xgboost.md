---
title: XGBoost
description: W&B で ツリー を追跡します。
menu:
  default:
    identifier: ja-guides-integrations-xgboost
    parent: integrations
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

`wandb` ライブラリには、XGBoost でのトレーニングからメトリクス、config、および保存されたブースターを記録するための `WandbCallback` コールバックがあります。ここでは、XGBoost `WandbCallback` からの出力を含む、**[ライブの Weights & Biases ダッシュボード](https://wandb.ai/morg/credit_scorecard)** を確認できます。

{{< img src="/images/integrations/xgb_dashboard.png" alt="Weights & Biases dashboard using XGBoost" >}}

## 始め方

XGBoost のメトリクス、config、およびブースターモデルを Weights & Biases に記録するには、`WandbCallback` を XGBoost に渡すだけです。

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# Start a wandb run
run = wandb.init()

# Pass WandbCallback to the model
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# Close your wandb run
run.finish()
```

XGBoost と Weights & Biases でのロギングの詳細については、**[この notebook](https://wandb.me/xgboost)** を開いてください。

## `WandbCallback` リファレンス

### 機能
`WandbCallback` を XGBoost モデルに渡すと、次のようになります。
- ブースターモデルの構成を Weights & Biases に記録します。
- XGBoost によって収集された評価 メトリクス (rmse、accuracy など) を Weights & Biases に記録します。
- XGBoost によって収集されたトレーニング メトリクス (eval_set にデータを提供する場合) を記録します。
- 最高のスコアと最高のイテレーションを記録します。
- トレーニング済みのモデルを保存して Weights & Biases Artifacts にアップロードします (`log_model = True` の場合)。
- `log_feature_importance=True` (デフォルト) の場合、特徴量のインポータンスプロットを記録します。
- `define_metric=True` (デフォルト) の場合、`wandb.summary` で最適な評価 メトリクスをキャプチャします。

### 引数
- `log_model`: (boolean) True の場合、モデルを保存して Weights & Biases Artifacts にアップロードします。

- `log_feature_importance`: (boolean) True の場合、特徴量のインポータンス棒グラフを記録します。

- `importance_type`: (str) ツリー モデルの場合は `{weight, gain, cover, total_gain, total_cover}` のいずれか。線形モデルの場合は weight。

- `define_metric`: (boolean) True (デフォルト) の場合、`wandb.summary` でトレーニングの最後のステップではなく、最適なステップでのモデルのパフォーマンスをキャプチャします。

[WandbCallback のソース コード](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)を確認できます。

その他の例については、[GitHub の examples リポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)を確認してください。

## Sweeps でハイパーパラメータを チューニングする

モデルのパフォーマンスを最大限に引き出すには、ツリーの深さや学習率などのハイパーパラメータを チューニングする必要があります。Weights & Biases には、大規模なハイパーパラメータ テスト実験を構成、調整、および分析するための強力な ツールキットである [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) が含まれています。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

この [XGBoost & Sweeps Python スクリプト](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)を試すこともできます。

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="Summary: trees outperform linear learners on this classification dataset." >}}
