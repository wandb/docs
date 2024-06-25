---
description: ツリーをW&Bでトラッキングしよう。
displayed_sidebar: default
---


# LightGBM

`wandb`ライブラリには[LightGBM](https://lightgbm.readthedocs.io/en/latest/)向けの特別なコールバックが含まれています。また、Weights & Biasesの汎用的なログ機能を使って、ハイパーパラメーター探索のような大規模な実験をトラッキングするのも簡単です。

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# メトリクスをW&Bにログする
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 特徴インポータンスプロットをログし、モデルチェックポイントをW&Bにアップロードする
log_summary(gbm, save_model_checkpoint=True)
```

:::info
動作するコード例をお探しですか？[GitHubの例のリポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)を確認するか、[Colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple\_LightGBM\_Integration.ipynb)を試してみてください。
:::

## Sweepsでハイパーパラメーターをチューニングする

モデルの最大性能を引き出すためには、ツリーの深さや学習率などのハイパーパラメーターをチューニングする必要があります。Weights & Biasesには、[Sweeps](../sweeps/)という強力なツールキットが含まれており、大規模なハイパーパラメーターテスト実験の設定、オーケストレーション、分析を行うことができます。

:::info
これらのツールについて詳しく知り、XGBoostとSweepsを使う例を見たい方は、[このインタラクティブなColabノートブック](http://wandb.me/xgb-sweeps-colab)をご覧ください。
:::

![tl;dr: この分類データセットでは、ツリーが線形学習者を上回るパフォーマンスを発揮します。](/images/integrations/lightgbm_sweeps.png)