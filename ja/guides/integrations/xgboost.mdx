---
title: XGBoost
description: ツリーを W&B でトラッキングしましょう。
menu:
  default:
    identifier: ja-guides-integrations-xgboost
    parent: integrations
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

`wandb` ライブラリには、XGBoost のトレーニングからメトリクス、設定、保存されたブースターをログするための `WandbCallback` コールバックがあります。ここでは、XGBoost `WandbCallback` の出力を含む **[ライブ Weights & Biases ダッシュボード](https://wandb.ai/morg/credit_scorecard)** を確認できます。

{{< img src="/images/integrations/xgb_dashboard.png" alt="Weights & Biases ダッシュボードを使用した XGBoost" >}}

## 始めに

XGBoost で収集したメトリクス、設定、ブースターモデルを Weights & Biases にログするのは、XGBoost に `WandbCallback` を渡すだけで簡単です。

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb run を開始
run = wandb.init()

# モデルに WandbCallback を渡す
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# wandb run を終了
run.finish()
```

**[このノートブック](https://wandb.me/xgboost)** を開いて、XGBoost と Weights & Biases を使用したログの詳細な方法を見ることができます。

## `WandbCallback` リファレンス

### 機能
`WandbCallback` を XGBoost モデルに渡すと、以下のことが行えます:
- ブースターモデルの設定を Weights & Biases にログする
- XGBoost によって収集された評価メトリクス（例: rmse, accuracy）を Weights & Biases にログする
- XGBoost で収集されたトレーニングメトリクスをログする（eval_set にデータを提供する場合）
- 最良のスコアと最良のイテレーションをログする
- トレーニング済みモデルを Weights & Biases Artifacts に保存およびアップロードする（`log_model = True` の場合）
- `log_feature_importance=True`（デフォルト）の場合、特徴重要度のプロットをログする
- `define_metric=True`（デフォルト）の場合、`wandb.summary` に最良の評価メトリックをキャプチャする

### 引数
- `log_model`: (boolean) True の場合、モデルを Weights & Biases Artifacts に保存しアップロードする

- `log_feature_importance`: (boolean) True の場合、特徴重要度の棒グラフをログする

- `importance_type`: (str) `{weight, gain, cover, total_gain, total_cover}` のいずれかでツリーモデルに適用。重みは線形モデルに対応。

- `define_metric`: (boolean) True（デフォルト）の場合、トレーニングの最良のステップでモデルのパフォーマンスを `wandb.summary` にキャプチャする（最後のステップではなく）。

`WandbCallback` の[ソースコード](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)を確認できます。

追加の例は、[GitHub の例のリポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)をチェックしてください。

## Sweep でハイパーパラメーターをチューニングする

モデルの最大パフォーマンスを引き出すには、ツリーの深さや学習率など、ハイパーパラメーターをチューニングする必要があります。Weights & Biases には、大規模なハイパーパラメーターテスト実験を設定、編成、分析するための強力なツールキットである [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) が含まれています。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

この [XGBoost & Sweeps Python スクリプト](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py) も試すことができます。

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="要約: この分類データセットではツリーが線形学習者を上回る。" >}}