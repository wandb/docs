---
title: Catalyst
description: Catalyst（PyTorch フレームワーク）で W&B を統合する方法
menu:
  default:
    identifier: ja-guides-integrations-catalyst
    parent: integrations
weight: 30
---

[Catalyst](https://github.com/catalyst-team/catalyst) は、再現性、迅速な実験、コードベースの再利用に焦点をあてた PyTorch フレームワークで、ディープラーニングの R&D を効率化し、新しいものを生み出す手助けをします。

Catalyst には、パラメータ、メトリクス、画像、その他のアーティファクトをログするための W&B インテグレーションが含まれています。

Python や Hydra を使った例も含む[インテグレーションのドキュメント](https://catalyst-team.github.io/catalyst/api/loggers.html#catalyst.loggers.wandb.WandbLogger)もぜひご覧ください。

## インタラクティブな例

Catalyst と W&B のインテグレーションを実際に体験するには、[こちらの colab の例](https://colab.research.google.com/drive/1PD0LnXiADCtt4mu7bzv7VfQkFXVrPxJq?usp=sharing) を実行してみてください。