---
title: Catalyst
description: Catalyst、PyTorch フレームワークに W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-catalyst
    parent: integrations
weight: 30
---

[Catalyst](https://github.com/catalyst-team/catalyst) は、再現性、迅速な実験、およびコードベースの再利用に焦点を当てたディープ ラーニング R&D のための PyTorch フレームワークです。これにより、新しいものを創り出すことができます。

Catalyst には、パラメータ、メトリクス、画像、その他のアーティファクトをログするための W&B インテグレーションが含まれています。

Python と Hydra を使用した例を含むインテグレーションの [ドキュメント](https://catalyst-team.github.io/catalyst/api/loggers.html#catalyst.loggers.wandb.WandbLogger) をチェックしてください。

## インタラクティブな例

Catalyst と W&B インテグレーションを実際に見るために、[Colab の例](https://colab.research.google.com/drive/1PD0LnXiADCtt4mu7bzv7VfQkFXVrPxJq?usp=sharing) を実行してください。