---
title: XGBoost
description: W&B で ツリー を追跡しましょう。
menu:
  default:
    identifier: ja-guides-integrations-xgboost
    parent: integrations
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

`wandb` ライブラリには、XGBoost を使用したトレーニングからメトリクス、config、および保存されたブースターを記録するための `WandbCallback` コールバックがあります。ここでは、XGBoost の `WandbCallback` からの出力を備えた、**[ライブの Weights & Biases ダッシュボード](https://wandb.ai/morg/credit_scorecard)** を見ることができます。

{{< img src="/images/integrations/xgb_dashboard.png" alt="XGBoost を使用した Weights & Biases ダッシュボード" >}}

## はじめに

XGBoost のメトリクス、config、およびブースターモデルを Weights & Biases に記録するのは、`WandbCallback` を XGBoost に渡すのと同じくらい簡単です。

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb run を開始する
run = wandb.init()

# WandbCallback をモデルに渡す
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# wandb run を終了する
run.finish()
```

XGBoost および Weights & Biases を使用したログ記録の詳細については、**[この notebook](https://wandb.me/xgboost)** を開いてください。

## `WandbCallback` リファレンス

### 機能
`WandbCallback` を XGBoost モデルに渡すと、次のようになります。
- ブースターモデルの設定を Weights & Biases に記録する
- rmse、精度など、XGBoost によって収集された評価メトリクスを Weights & Biases に記録する
- XGBoost によって収集されたトレーニングメトリクスを記録する（eval_set にデータを提供する場合）
- ベストスコアとベストイテレーションを記録する
- トレーニング済みのモデルを保存して Weights & Biases Artifacts にアップロードする（`log_model = True` の場合）
- `log_feature_importance=True` （デフォルト）の場合、特徴量のインポータンスプロットを記録します。
- `define_metric=True` （デフォルト）の場合、`wandb.summary` でトレーニングの最後のステップではなく、最適なステップでモデルの評価メトリクスをキャプチャします。

### 引数
- `log_model`: (boolean) True の場合、モデルを保存して Weights & Biases Artifacts にアップロードします

- `log_feature_importance`: (boolean) True の場合、特徴量のインポータンス棒グラフを記録します

- `importance_type`: (str) ツリーモデルの場合は `{weight, gain, cover, total_gain, total_cover}` のいずれか。線形モデルの場合は weight。

- `define_metric`: (boolean) True (デフォルト) の場合、`wandb.summary` でトレーニングの最後のステップではなく、最適なステップでモデルのパフォーマンスをキャプチャします。

[WandbCallback のソースコード](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)を確認できます。

その他の例については、[GitHub の examples リポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)を確認してください。

## Sweeps でハイパーパラメータをチューニングする

モデルから最大のパフォーマンスを得るには、ツリーの深さや学習率などのハイパーパラメータをチューニングする必要があります。Weights & Biases には、大規模なハイパーパラメータテスト実験を構成、調整、および分析するための強力なツールキットである [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) が含まれています。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

この [XGBoost & Sweeps Python スクリプト](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)を試すこともできます。

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="概要: この分類データセットでは、ツリーが線形学習器よりも優れています。" >}}
