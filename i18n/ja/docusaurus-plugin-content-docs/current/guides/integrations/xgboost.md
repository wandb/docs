---
description: Track your trees with W&B.
displayed_sidebar: ja
---

# XGBoost

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://wandb.me/xgboost)

`wandb`ライブラリには、XGBoostでのトレーニング時にメトリクス、設定、そして保存されたブースターをログに記録するための`WandbCallback`コールバックがあります。ここでは、XGBoostの`WandbCallback`の出力が含まれる**[Weights & Biasesのライブダッシュボード](https://wandb.ai/morg/credit_scorecard)**をご覧いただけます。

![XGBoostを使用したWeights & Biasesダッシュボード](/images/integrations/xgb_dashboard.png)

## はじめに

Weights & BiasesへのXGBoostメトリクス、設定、およびブースターモデルのログ記録は、`WandbCallback`をXGBoostに渡すだけで簡単に行えます。

```python
from wandb.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb runを開始
run = wandb.init()

# WandbCallbackをモデルに渡す
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# wandb runを終了
run.finish()
```
**[このノートブック](https://wandb.me/xgboost)** を開いて、XGBoostとWeights＆Biasesを使ったログの詳細を確認してください。

## WandbCallback

### 機能
`WandbCallback`をXGBoostモデルに渡すと、次のことができます。
- ブースターモデルの設定をWeights＆Biasesにログ
- XGBoostによって収集される評価指標（rmse、精度など）をWeights＆Biasesにログ
- XGBoostによって収集されるトレーニング指標をログ（eval_setにデータを提供する場合）
- 最高スコアと最適なイテレーションをログ
- トレーニング済みモデルをWeights＆Biases Artifactsに保存してアップロード（`log_model = True` の場合）
- `log_feature_importance=True`（デフォルト）で特徴量重要度プロットをログ
- `define_metric=True`（デフォルト）で、`wandb.summary` に最適なステップでのモデルのパフォーマンスを記録

### 引数
`log_model`: (boolean) Trueの場合、モデルをWeights＆Biases Artifactsに保存してアップロード

`log_feature_importance`: (boolean) Trueの場合、特徴量重要度の棒グラフをログ

`importance_type`: (str) 木モデルの場合、{weight, gain, cover, total_gain, total_cover} のいずれか。線形モデルの場合、weight。

`define_metric`: (boolean) True（デフォルト）の場合、トレーニングの最後のステップではなく、最適なステップでのモデルのパフォーマンスを `wandb.summary` に記録。

WandbCallbackのソースコードは[こちら](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)で見ることができます。

:::info
動作するコード例をもっと探していますか？[GitHubの弊社リポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)で例をチェックするか、[Colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit\_Scorecards\_with\_XGBoost\_and\_W%26B.ipynb)をお試しください。
:::
## スイープを使ったハイパーパラメータチューニング

モデルの最大性能を達成するためには、ハイパーパラメーター（例: 木の深さや学習率）をチューニングする必要があります。Weights & Biasesには[Sweeps](../sweeps/)が含まれており、大規模なハイパーパラメータテスト実験の設定、制御、解析に適したパワフルなツールキットです。

:::info

これらのツールについて詳しく学ぶために、XGBoostとSweepsを使った例を見るには、[このインタラクティブなColabノートブック](http://wandb.me/xgb-sweeps-colab)をチェックしてみてください。また、XGBoostとSweepsの[Pythonスクリプト](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost\_tune.py)も試してみてください。

:::

![要約：この分類データセットでは、木が線形学習者よりも優れた性能を発揮します。](/images/integrations/xgboost_sweeps_example.png)