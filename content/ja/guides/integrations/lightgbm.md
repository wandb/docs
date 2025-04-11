---
title: LightGBM
description: W&B でツリーをトラッキングする。
menu:
  default:
    identifier: ja-guides-integrations-lightgbm
    parent: integrations
weight: 190
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb" >}}

`wandb` ライブラリには、[LightGBM](https://lightgbm.readthedocs.io/en/latest/) 用の特別なコールバックが含まれています。また、Weights & Biases の一般的なログ機能を使用して、大規模な実験やハイパーパラメータ探索を追跡することも簡単です。

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# メトリクスを W&B にログ
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 特徴量のインポータンスプロットをログし、モデルのチェックポイントを W&B にアップロード
log_summary(gbm, save_model_checkpoint=True)
```

{{% alert %}}
動作するコード例をお探しですか？[GitHub の例のリポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)をチェックしてください。
{{% /alert %}}

## ハイパーパラメーターの調整と Sweeps

モデルから最大のパフォーマンスを引き出すには、ツリーの深さや学習率のようなハイパーパラメーターを調整する必要があります。Weights & Biases は、大規模なハイパーパラメータのテスト実験を設定、調整、分析するための強力なツールキットである [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})を含んでいます。

これらのツールについて学び、XGBoost で Sweeps を使用する方法の例を確認するには、この対話型 Colab ノートブックをチェックしてください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

{{< img src="/images/integrations/lightgbm_sweeps.png" alt="要約: この分類データセットでツリーは線形学習者を上回る。" >}}