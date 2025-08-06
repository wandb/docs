---
title: XGBoost
description: W&B でツリーをトラッキングしましょう。
menu:
  default:
    identifier: xgboost
    parent: integrations
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

`wandb` ライブラリには、XGBoost でのトレーニング時にメトリクスや設定、保存されたブースターを記録できる `WandbCallback` コールバックがあります。下記のリンクから [ライブ W&B ダッシュボード](https://wandb.ai/morg/credit_scorecard) で、XGBoost の `WandbCallback` からの出力例を見ることができます。

{{< img src="/images/integrations/xgb_dashboard.png" alt="XGBoost を使った W&B ダッシュボード" >}}

## はじめよう

XGBoost のメトリクス・設定・ブースターモデルを W&B に記録するには、単純に `WandbCallback` を XGBoost に渡すだけでOKです。

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb run を開始
with wandb.init() as run:
  # モデルに WandbCallback を渡す
  bst = XGBClassifier()
  bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])
```

より詳しく XGBoost と W&B でのロギングを見たい場合は[こちらのノートブック](https://wandb.me/xgboost)をご覧ください。

## `WandbCallback` リファレンス

### 機能
XGBoost モデルに `WandbCallback` を渡すと以下のことが行われます:
- ブースターモデルの設定を W&B に記録
- XGBoost で取得された評価メトリクス（例: rmse, accuracy など）を W&B に記録
- eval_set を提供した場合、XGBoost で取得したトレーニングメトリクスを記録
- ベストスコアとベストイテレーションを記録
- トレーニングしたモデルを W&B Artifacts に保存・アップロード（`log_model = True` の場合）
- `log_feature_importance=True`（デフォルト時）に特徴量インポータンスプロットを記録
- `define_metric=True`（デフォルト時）は `wandb.Run.summary` にベスト評価メトリクスを取得

### 引数
- `log_model`: (ブール値) True の場合、モデルを W&B Artifacts に保存・アップロードします

- `log_feature_importance`: (ブール値) True の場合、特徴量インポータンスの棒グラフを記録します

- `importance_type`: (文字列) ツリーモデルの場合は `{weight, gain, cover, total_gain, total_cover}` のいずれか。線形モデルの場合は weight。

- `define_metric`: (ブール値) True（デフォルト）の場合、トレーニングのラストステップではなくベストステップでのモデル性能を `run.summary` に記録します

`WandbCallback` の[ソースコードはこちら](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)から確認できます。

追加のサンプルは、[GitHub の examples リポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)もご覧ください。

## Sweeps でハイパーパラメータをチューニングしよう

モデルのパフォーマンスを最大化するには、ツリーの深さや学習率といったハイパーパラメータのチューニングが重要です。W&B の [Sweeps]({{< relref "/guides/models/sweeps/" >}}) は、大規模なハイパーパラメータテスト実験を構成・実行・解析するための強力なツールキットです。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

また、[XGBoost & Sweeps Python スクリプト](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py) もぜひお試しください。

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="XGBoost パフォーマンス比較" >}}