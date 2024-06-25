---
description: W&Bでツリーをトラッキングしよう。
displayed_sidebar: default
---


# XGBoost

[**Try in a Colab Notebook here →**](https://wandb.me/xgboost)

`wandb` ライブラリには、XGBoostを使用したトレーニングからメトリクス、設定、保存されたブースターをログに記録するための `WandbCallback` コールバックがあります。ここでは、XGBoostの `WandbCallback` の出力を含む **[ライブ Weights & Biases ダッシュボード](https://wandb.ai/morg/credit_scorecard)** を見ることができます。

![Weights & Biases ダッシュボード using XGBoost](/images/integrations/xgb_dashboard.png)

## はじめに

XGBoostのメトリクス、設定、ブースターモデルをWeights & Biasesにログとして記録するのは、`WandbCallback` をXGBoostに渡すだけで簡単にできます：

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb run を開始
run = wandb.init()

# モデルにWandbCallbackを渡す
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# wandb run を終了
run.finish()
```

XGBoostとWeights & Biasesでのログ記録について詳しく知りたい方は、**[このノートブック](https://wandb.me/xgboost)** をご覧ください。

## WandbCallback

### 機能
`WandbCallback` をXGBoostモデルに渡すことで：
- ブースターモデルの設定をWeights & Biasesにログします
- XGBoostによって収集された評価メトリクス（例えばrmse、accuracyなど）をWeights & Biasesにログします
- XGBoostによって収集されたトレーニングメトリクス（eval_setにデータを提供した場合）をログします
- 最良のスコアと最良のイテレーションをログします
- トレーニング済みモデルをWeights & Biases Artifactsに保存・アップロードします（`log_model = True` の場合）
- `log_feature_importance=True`（デフォルト）の場合、特徴量のインポータンスプロットをログします
- `define_metric=True`（デフォルト）の場合、`wandb.summary` に最良の評価メトリクスをキャプチャします

### 引数
`log_model`: (boolean) Trueの場合、モデルを保存してWeights & Biases Artifactsにアップロードします

`log_feature_importance`: (boolean) Trueの場合、特徴量のインポータンスバープロットをログします

`importance_type`: (str) {weight, gain, cover, total_gain, total_cover} のいずれか（ツリーモデルの場合）。線形モデルの場合はweight。

`define_metric`: (boolean) True（デフォルト）の場合、トレーニングの最終ステップではなく最良のステップでのモデル性能を `wandb.summary` にキャプチャします

WandbCallback のソースコードは [こちら](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py) で見つけることができます。

:::info
さらに動作するコード例を探している方は、[GitHubの例のリポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms) をチェックするか、[Colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit\_Scorecards\_with\_XGBoost\_and\_W%26B.ipynb) を試してみてください。
:::

## Sweepsを使ったハイパーパラメータのチューニング

モデルの最大性能を引き出すには、ツリーの深さや学習率などのハイパーパラメータをチューニングする必要があります。Weights & Biases では、大規模なハイパーパラメータテストの実験を設定、調整、分析するための強力なツールキットである [Sweeps](../sweeps/) を提供しています。

:::info
これらのツールについてさらに学び、XGBoostとSweepsを使った例を見てみたい方は、[このインタラクティブなColabノートブック](http://wandb.me/xgb-sweeps-colab) をチェックするか、XGBoost & Sweeps の[このpythonスクリプト](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost\_tune.py) を試してください。
:::

![tl;dr: trees outperform linear learners on this classification dataset.](/images/integrations/xgboost_sweeps_example.png)
