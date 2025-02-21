---
title: Catalyst
description: Pytorch の フレームワーク である Catalyst に W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-catalyst
    parent: integrations
weight: 30
---

[Catalyst](https://github.com/catalyst-team/catalyst) は、再現性 、迅速な 実験 、および コード ベースの 再利用に 重点を置いた ディープラーニング R&D 用の PyTorch framework であり、新しいものを 作成できます。

Catalyst には、 パラメータ 、 メトリクス 、画像、および その他の Artifacts を ログ記録するための W&B integration が 含まれています。

Python と Hydra を 使用した 例が 含まれている、 [integration の ドキュメント](https://catalyst-team.github.io/catalyst/api/loggers.html#catalyst.loggers.wandb.WandbLogger) を ご覧ください。

## Interactive Example

Catalyst と W&B integration の 動作を 確認するには、[example colab](https://colab.research.google.com/drive/1PD0LnXiADCtt4mu7bzv7VfQkFXVrPxJq?usp=sharing) を 実行してください。
