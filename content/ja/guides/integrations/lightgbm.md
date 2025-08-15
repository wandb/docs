---
title: LightGBM
description: W&B でツリーをトラッキングしましょう。
menu:
  default:
    identifier: ja-guides-integrations-lightgbm
    parent: integrations
weight: 190
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb" >}}

`wandb` ライブラリには、[LightGBM](https://lightgbm.readthedocs.io/en/latest/) 専用のコールバックが含まれています。また、W&B の汎用的なログ機能を使えば、ハイパーパラメーター探索のような大規模な実験も簡単にトラッキングできます。

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# メトリクスを W&B に記録
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 特徴量のインポータンスプロットを記録し、モデルのチェックポイントも W&B にアップロード
log_summary(gbm, save_model_checkpoint=True)
```

{{% alert %}}
実際に動かせるコード例を探していますか？ [GitHub のサンプルリポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms) をチェックしてみてください。
{{% /alert %}}

## Sweeps でハイパーパラメーターをチューニングする

モデルの性能を最大限に引き出すためには、ツリーの深さや学習率などのハイパーパラメーターのチューニングが必要です。W&B の [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は、大規模なハイパーパラメーター探索実験を設定・制御・分析するための強力なツールキットです。

これらのツールについてさらに知りたい方や、Sweeps を XGBoost で利用する具体例を見たい方は、以下のインタラクティブな Colabノートブック をご覧ください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

{{< img src="/images/integrations/lightgbm_sweeps.png" alt="LightGBM パフォーマンス比較" >}}