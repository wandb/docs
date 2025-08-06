---
title: Catalyst
description: Catalyst（PyTorch フレームワーク）で W&B を統合する方法
menu:
  default:
    identifier: catalyst
    parent: integrations
weight: 30
---

[Catalyst](https://github.com/catalyst-team/catalyst) は、再現性、迅速な実験、コードベースの再利用に重点を置いた PyTorch のディープラーニング R&D フレームワークです。これにより新しいものをスムーズに作成できます。

Catalyst には、パラメータ、メトリクス、画像、その他のアーティファクトをログするための W&B インテグレーションが含まれています。

Python や Hydra を使った例を含め、[インテグレーションに関する公式ドキュメント](https://catalyst-team.github.io/catalyst/api/loggers.html#catalyst.loggers.wandb.WandbLogger)もご覧ください。

## インタラクティブな例

Catalyst と W&B のインテグレーションを体験できる[Colab の例](https://colab.research.google.com/drive/1PD0LnXiADCtt4mu7bzv7VfQkFXVrPxJq?usp=sharing)を実行してみてください。