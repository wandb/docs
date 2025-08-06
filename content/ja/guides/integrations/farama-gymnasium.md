---
title: Farama Gymnasium
description: W&B を Farama Gymnasium と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-farama-gymnasium
    parent: integrations
weight: 90
---

[Farama Gymnasium](https://gymnasium.farama.org/#) をご利用の場合、`gymnasium.wrappers.Monitor` で生成された環境の動画を自動でログします。`monitor_gym` というキーワード引数を [`wandb.init`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) で `True` に設定するだけでOKです。

W&B の gymnasium インテグレーションはとてもシンプルです。`gymnasium` からログされた [動画ファイル名を確認](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67) し、その名前で登録するか、見つからなければ `"videos"` という名前で保存します。もっと細かく管理したい場合は、[動画を手動でログする]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) こともできます。

Gymnasium を CleanRL ライブラリと一緒に使う方法については、この [Report](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw) をご覧ください。

{{< img src="/images/integrations/gymnasium.png" alt="Gymnasium RL environment" >}}