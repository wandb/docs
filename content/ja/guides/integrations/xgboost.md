---
title: XGBoost
description: W&B で ツリーを追跡しましょう。
menu:
  default:
    identifier: ja-guides-integrations-xgboost
    parent: integrations
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

`wandb` ライブラリには、XGBoost のトレーニングからメトリクスや設定、保存済み booster をログするための `WandbCallback` コールバックがあります。ここでは、XGBoost の `WandbCallback` の出力を表示する [ライブの W&B Dashboard](https://wandb.ai/morg/credit_scorecard) を確認できます。

{{< img src="/images/integrations/xgb_dashboard.png" alt="XGBoost を使用した W&B Dashboard" >}}

## はじめに

XGBoost のメトリクス、設定、booster モデルを W&B にログするには、`WandbCallback` を XGBoost に渡すだけで OK です:

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb の run を開始
with wandb.init() as run:
  # モデルに WandbCallback を渡す
  bst = XGBClassifier()
  bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])
```

XGBoost と W&B でのロギングを詳しく知るには、[このノートブック](https://wandb.me/xgboost) をご覧ください。

## `WandbCallback` リファレンス

### 機能
`WandbCallback` を XGBoost モデルに渡すと、次のことを行います:
- booster モデルの設定を W&B にログする
- XGBoost が収集する rmse、accuracy などの評価メトリクスを W&B にログする
- XGBoost が収集するトレーニングメトリクスをログする（eval_set にデータを渡した場合）
- ベストスコアとベストイテレーションをログする
- 学習済みモデルを W&B Artifacts に保存・アップロードする（`log_model = True` のとき）
- `log_feature_importance=True`（デフォルト）の場合、特徴量のインポータンスプロットをログする
- `define_metric=True`（デフォルト）の場合、`wandb.Run.summary` にベスト評価メトリクスを記録する

### 引数
- `log_model`: (boolean) True の場合、モデルを W&B Artifacts に保存してアップロードします

- `log_feature_importance`: (boolean) True の場合、特徴量のインポータンスの棒グラフをログします

- `importance_type`: (str) ツリーモデルでは `{weight, gain, cover, total_gain, total_cover}` のいずれか。線形モデルでは weight。

- `define_metric`: (boolean) True（デフォルト）の場合、トレーニングの最後のステップではなくベストステップでのモデル性能を `run.summary` に記録します。


[`WandbCallback` のソースコード](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py) を参照できます。

さらに例が必要な場合は、[GitHub 上のサンプル集リポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms) をチェックしてください。

## Sweeps でハイパーパラメーターをチューニングする

モデルの性能を最大化するには、ツリーの深さや学習率のようなハイパーパラメーターをチューニングする必要があります。W&B [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は、大規模なハイパーパラメーター探索実験を設定・オーケストレーション・分析するための強力なツールキットです。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

こちらの [XGBoost & Sweeps の Python スクリプト](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py) もお試しください。

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="XGBoost の性能比較" >}}