---
title: Catalyst
description: PyTorch フレームワークである Catalyst に W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-catalyst
    parent: integrations
weight: 30
---

[Catalyst](https://github.com/catalyst-team/catalyst) は PyTorch フレームワークで、ディープラーニング R&D における再現性、迅速な実験、コードベースの再利用に重点を置き、新しいものを生み出せるようにします。

Catalyst には、パラメータ、メトリクス、画像、その他の Artifacts をログする W&B インテグレーションが含まれています。

Python と Hydra を使った例を含む、[インテグレーションのドキュメント](https://catalyst-team.github.io/catalyst/api/loggers.html#catalyst.loggers.wandb.WandbLogger) を参照してください。

## インタラクティブな例

Catalyst と W&B のインテグレーションが動作する様子を見るには、[サンプルの Colab](https://colab.research.google.com/drive/1PD0LnXiADCtt4mu7bzv7VfQkFXVrPxJq?usp=sharing) を実行してください。