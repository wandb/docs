---
title: LightGBM
description: W&B であなたの ツリー をトラッキングしましょう。
menu:
  default:
    identifier: lightgbm
    parent: integrations
weight: 190
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb" >}}

`wandb` ライブラリには、[LightGBM](https://lightgbm.readthedocs.io/en/latest/) 用の特別なコールバックが含まれています。また、W&B の一般的なログ機能を使って、大規模な実験やハイパーパラメーター探索も簡単に記録できます。

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# W&B にメトリクスをログします
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 特徴量のインポータンスプロットをログし、モデルのチェックポイントを W&B にアップロードします
log_summary(gbm, save_model_checkpoint=True)
```

{{% alert %}}
実際に動作するコード例をお探しですか？[GitHub 上のサンプル集](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)をご覧ください。
{{% /alert %}}

## Sweeps を使ったハイパーパラメーターのチューニング

モデルの性能を最大限に引き出すためには、ツリーの深さや学習率などのハイパーパラメーターの最適化が不可欠です。W&B の [Sweeps]({{< relref "/guides/models/sweeps/" >}}) は、大規模なハイパーパラメーターテストの実験を構成・管理・分析できる強力なツールキットです。

これらのツールについてさらに知りたい方、XGBoost で Sweeps を活用する例を見たい方は、ぜひインタラクティブな Colab ノートブックをチェックしてください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

{{< img src="/images/integrations/lightgbm_sweeps.png" alt="LightGBM の性能比較" >}}