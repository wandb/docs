---
title: XGBoost
description: W&B でツリーをトラッキングしましょう。
menu:
  default:
    identifier: ja-guides-integrations-xgboost
    parent: integrations
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

`wandb` ライブラリには、XGBoost トレーニング時にメトリクスや設定、保存済み booster を記録できる `WandbCallback` コールバックがあります。こちらの [ライブ W&B Dashboard](https://wandb.ai/morg/credit_scorecard) で、XGBoost の `WandbCallback` から出力された例を見ることができます。

{{< img src="/images/integrations/xgb_dashboard.png" alt="XGBoost を使った W&B Dashboard" >}}

## はじめる

XGBoost のメトリクス、設定、booster モデルを W&B に記録するのはとても簡単で、`WandbCallback` を XGBoost に渡すだけです。

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

[XGBoost と W&B でのロギングの詳細なノートブック](https://wandb.me/xgboost)もご覧いただけます。

## `WandbCallback` リファレンス

### 機能
XGBoost モデルに `WandbCallback` を渡すことで、以下のことが自動で行われます。

- booster モデルの設定情報を W&B に記録
- XGBoost によって計算される rmse や accuracy などの評価メトリクスを W&B に記録
- XGBoost で収集されるトレーニングメトリクスを記録（`eval_set` にデータを与えた場合）
- ベストスコアとそのイテレーションを記録
- 学習済みモデルを W&B Artifacts へ保存・アップロード（`log_model=True` 時）
- `log_feature_importance=True`（デフォルト）で特徴量インポータンスプロットを記録
- `define_metric=True`（デフォルト）の場合、`wandb.Run.summary` にベスト評価メトリクスを記録

### 引数
- `log_model`: （bool値）True の場合、モデルを W&B Artifacts に保存・アップロードします

- `log_feature_importance`: （bool値）True の場合、特徴量インポータンスのバー プロットを記録

- `importance_type`: （str）ツリーモデルにおいては `{weight, gain, cover, total_gain, total_cover}` のいずれか。線形モデルの場合は weight。

- `define_metric`: （bool値）True（デフォルト）の場合、トレーニングの最後ではなく、最良ステップ時のモデル性能を `run.summary` に記録


`WandbCallback` の[ソースコードはこちら](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)で確認できます。

他の例については、[GitHub の例リポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)もご覧ください。

## Sweeps でハイパーパラメータをチューニング

最大限のモデル性能を引き出すには、ツリーの深さや学習率などのハイパーパラメータをチューニングする必要があります。W&B の [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は、ハイパーパラメータのテスト実験を構成・管理・解析するための強力なツールキットです。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

[XGBoost & Sweeps の Python スクリプト](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py) もお試しいただけます。

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="XGBoost パフォーマンス比較" >}}