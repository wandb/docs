---
title: LightGBM
description: W&B で ツリー を追跡しましょう。
menu:
  default:
    identifier: ja-guides-integrations-lightgbm
    parent: integrations
weight: 190
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb" >}}

`wandb` ライブラリには、[LightGBM](https://lightgbm.readthedocs.io/en/latest/) 用の特別な callback が含まれています。Weights & Biases の汎用的なログ機能を使用すると、ハイパーパラメーター探索などの大規模な experiment を簡単に追跡できます。

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# W&B にメトリクスを記録
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 特徴量のインポータンスプロットを記録し、モデルのチェックポイントを W&B にアップロード
log_summary(gbm, save_model_checkpoint=True)
```

{{% alert %}}
動作するコード例をお探しですか？[GitHub の examples リポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms) をご覧ください。
{{% /alert %}}

## Sweeps でハイパーパラメーターを調整する

モデルのパフォーマンスを最大限に引き出すには、木の深さや学習率などのハイパーパラメーターを調整する必要があります。Weights & Biases には、大規模なハイパーパラメーター テスト experiment の構成、調整、分析を行うための強力なツールキットである [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) が含まれています。

これらのツールについて詳しく知り、XGBoost で Sweeps を使用する方法の例については、このインタラクティブな Colabノートブック をご覧ください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

{{< img src="/images/integrations/lightgbm_sweeps.png" alt="Summary: trees outperform linear learners on this classification dataset." >}}
