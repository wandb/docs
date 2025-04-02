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

`wandb` ライブラリには、 [LightGBM](https://lightgbm.readthedocs.io/en/latest/) 用の特別な コールバック が含まれています。また、 Weights & Biases の汎用的な ログ 機能を使用すると、 ハイパーパラメーター 探索 のような大規模な 実験 を簡単に追跡できます。

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# メトリクス を W&B に ログ 記録
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 特徴量の インポータンスプロット を ログ 記録し、 モデル の チェックポイント を W&B にアップロード
log_summary(gbm, save_model_checkpoint=True)
```

{{% alert %}}
実際に動作する コード 例をお探しですか？ [GitHub の 例 のリポジトリ](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms) をご確認ください。
{{% /alert %}}

## Sweeps を使用した ハイパーパラメーター の調整

モデル のパフォーマンスを最大限に引き出すには、 ツリー の深さや学習率などの ハイパーパラメーター を調整する必要があります。Weights & Biases には、大規模な ハイパーパラメーター のテスト 実験 を構成、編成、分析するための強力な ツールキットである [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) が含まれています。

これらの ツール の詳細と、XGBoost で Sweeps を使用する方法の例については、このインタラクティブな Colab ノートブック をご覧ください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

{{< img src="/images/integrations/lightgbm_sweeps.png" alt="Summary: trees outperform linear learners on this classification dataset." >}}
