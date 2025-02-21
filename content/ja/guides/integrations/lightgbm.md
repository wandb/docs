---
title: LightGBM
description: W&B でツリーを追跡しましょう。
menu:
  default:
    identifier: ja-guides-integrations-lightgbm
    parent: integrations
weight: 190
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb" >}}

`wandb` ライブラリには、[LightGBM](https://lightgbm.readthedocs.io/en/latest/) 用の特別なコールバックが含まれています。また、Weights & Biases の一般的なロギング機能を使用して、大規模な実験 (例えば、ハイパーパラメーター探索) をトラッキングすることが簡単です。

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# メトリクスを W&B にログ
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 特徴量のインポータンスプロットをログし、モデルのチェックポイントを W&B にアップロード
log_summary(gbm, save_model_checkpoint=True)
```

{{% alert %}}
動作するコード例をお探しですか？ [GitHub の例のリポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)をチェックしてください。
{{% /alert %}}

## スイープでハイパーパラメーターを調整する

モデルの最大パフォーマンスを得るためには、ツリーの深さや学習率のようなハイパーパラメーターの調整が必要です。Weights & Biases には、大規模なハイパーパラメーターのテスト実験を設定・管理・解析するための強力なツールキットである [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) が含まれています。

これらのツールについて詳しく学び、XGBoost で Sweeps を使用する方法の例を見るには、この対話型の Colab ノートブックをチェックしてください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

{{< img src="/images/integrations/lightgbm_sweeps.png" alt="要約: この分類データセットでは、ツリーが線形学習者を上回っています。" >}}