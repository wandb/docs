---
title: Farama Gymnasium
description: W&B を Farama Gymnasium と統合する方法
menu:
  default:
    identifier: farama-gymnasium
    parent: integrations
weight: 90
---

もし [Farama Gymnasium](https://gymnasium.farama.org/#) を使っている場合、`gymnasium.wrappers.Monitor` で生成された環境の動画を自動的にログします。`monitor_gym` キーワード引数を [`wandb.init`]({{< relref "/ref/python/sdk/functions/init.md" >}}) に `True` として設定してください。

私たちの gymnasium インテグレーションはとてもシンプルです。[動画ファイル名](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67) を `gymnasium` から確認し、その名前で保存します。もし該当しない場合は `"videos"` という名前が使われます。もし詳しく管理したい場合は、いつでも手動で[動画をログする]({{< relref "/guides/models/track/log/media.md" >}}) こともできます。

Gymnasium を CleanRL ライブラリと一緒に使う方法については、この [レポート](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw) もご覧ください。

{{< img src="/images/integrations/gymnasium.png" alt="Gymnasium RL environment" >}}