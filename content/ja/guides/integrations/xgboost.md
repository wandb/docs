---
title: XGBoost
description: W&B でツリーを追跡しましょう。
menu:
  default:
    identifier: ja-guides-integrations-xgboost
    parent: integrations
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

`wandb` ライブラリには、XGBoost でのトレーニングからメトリクス、設定、保存されたブースターをログに記録するための `WandbCallback` コールバックがあります。ここでは、XGBoost の `WandbCallback` からの出力を含む **[ライブ Weights & Biases ダッシュボード](https://wandb.ai/morg/credit_scorecard)** を見ることができます。

{{< img src="/images/integrations/xgb_dashboard.png" alt="Weights & Biases ダッシュボード using XGBoost" >}}

## 始めてみましょう

XGBoost のメトリクス、設定、ブースターモデルを Weights & Biases にログ記録するのは、`WandbCallback` を XGBoost に渡すだけで簡単です。

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

**[このノートブック](https://wandb.me/xgboost)** を開いて、XGBoost と Weights & Biases でのログ記録について詳しく見ることができます。

## `WandbCallback` リファレンス

### 機能
`WandbCallback` を XGBoost モデルに渡すと以下のことが行われます:
- ブースターモデルの設定を Weights & Biases にログ記録
- XGBoost によって収集された評価メトリクス（例えば、rmse、精度など）を Weights & Biases にログ記録
- XGBoost によって収集されたトレーニングメトリクスをログ記録（eval_set にデータを提供した場合）
- 最良のスコアと最良のイテレーションをログ記録
- トレーニングされたモデルを Weights & Biases Artifacts に保存およびアップロード（`log_model = True` の場合）
- `log_feature_importance=True`（デフォルト）の場合に特徴量インポータンスプロットをログ記録
- `define_metric=True`（デフォルト）の場合に `wandb.summary` 内で最良の評価メトリクスをキャプチャ

### 引数
- `log_model`: (boolean) True の場合、モデルを保存し Weights & Biases Artifacts にアップロード

- `log_feature_importance`: (boolean) True の場合、特徴量インポータンスバーのプロットをログ記録

- `importance_type`: (str) `{weight, gain, cover, total_gain, total_cover}` のいずれか。ツリー モデルには weight、線形モデルには gain。

- `define_metric`: (boolean) True（デフォルト）の場合、トレーニングにおける最良のステップで、最後のステップではなく、`wandb.summary` にモデルの性能をキャプチャ 

`WandbCallback` の [ソースコード](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py) を確認できます。

追加の例については、 [GitHub の例のリポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)をチェックしてください。

## Sweeps でハイパーパラメーターをチューニングする

モデルから最大のパフォーマンスを得るには、ツリーの深さや学習率などのハイパーパラメーターをチューニングする必要があります。Weights & Biases には、大規模なハイパーパラメータ テスト実験を設定、調整、および分析するための強力なツールキットである [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) が含まれています。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

この [XGBoost & Sweeps Python スクリプト](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py) もお試しください。

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="要約: ツリーはこの分類データセットでの線形学習者を上回っています。" >}}