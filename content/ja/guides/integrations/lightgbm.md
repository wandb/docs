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

`wandb` ライブラリには [LightGBM](https://lightgbm.readthedocs.io/en/latest/) 向けの特別な コールバック が含まれています。さらに、W&B の汎用的な ログ 機能を使えば、ハイパーパラメーター探索のような大規模な 実験 を追跡するのも簡単です。

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# メトリクスを W&B にログする
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 特徴量のインポータンスプロットをログし、モデルのチェックポイントを W&B にアップロードする
log_summary(gbm, save_model_checkpoint=True)
```

{{% alert %}}
動くコード例をお探しですか？[GitHub 上のサンプル集リポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms) をご覧ください。
{{% /alert %}}

## Sweeps でハイパーパラメーターをチューニングする

モデルの性能を最大化するには、ツリー の深さや学習率などのハイパーパラメーターを調整する必要があります。W&B の [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) は、大規模なハイパーパラメーター検証 実験 の設定・オーケストレーション・分析を行える強力なツールキットです。

これらのツールの詳細や、XGBoost と Sweeps を組み合わせる例は、対話的な Colabノートブック をご覧ください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

{{< img src="/images/integrations/lightgbm_sweeps.png" alt="LightGBM の性能比較" >}}