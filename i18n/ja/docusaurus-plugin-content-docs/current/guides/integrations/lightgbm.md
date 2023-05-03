---
description: W&Bで木をトラッキングしましょう。
---

# LightGBM

`wandb`ライブラリには、[LightGBM](https://lightgbm.readthedocs.io/en/latest/)用の特別なコールバックが含まれています。また、Weights & Biasesの汎用的なログ機能を使って、ハイパーパラメータ探索のような大規模な実験を追跡することも簡単です。

```python
from wandb.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# W&Bにメトリクスをログ
gbm = lgb.train(..., callbacks=[wandb_callback()])

# W&Bに特徴量重要度図をログし、モデルのチェックポイントをアップロード
log_summary(gbm, save_model_checkpoint=True)
```

:::info
実行可能なコード例をお探しですか？[GitHubの例のリポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)を確認するか、[Colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple\_LightGBM\_Integration.ipynb)をお試しください。
:::

## スイープを使ってハイパーパラメータを調整

モデルの最大性能を引き出すには、木の深さや学習率などのハイパーパラメータを調整する必要があります。Weights & Biasesには、[スイープ](../sweeps/)という、大規模なハイパーパラメータテスト実験の設定、オーケストレーション、分析が可能な強力なツールキットが含まれています。

:::info
これらのツールの詳細や、XGBoostでスイープを使った例を見るには、[このインタラクティブなColabノートブック](http://wandb.me/xgb-sweeps-colab)をご覧ください。
:::
![要約：この分類データセットでは、木構造が線形学習よりも優れた性能を発揮します。](/images/integrations/lightgbm_sweeps.png)