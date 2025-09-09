---
title: Farama Gymnasium
description: W&B を Farama Gymnasium と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-farama-gymnasium
    parent: integrations
weight: 90
---

[Farama Gymnasium](https://gymnasium.farama.org/#) を使用している場合、`gymnasium.wrappers.Monitor` によって生成される 環境 の動画を自動でログします。[`wandb.init`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) の `monitor_gym` キーワード引数を `True` に設定するだけです。

Gymnasium とのインテグレーションはとても軽量です。`gymnasium` からログされる動画ファイルの [ファイル名を参照するだけ](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67) で、その名前をそのまま使用します。一致しない場合はフォールバックとして "videos" を使用します。より細かく制御したい場合は、手動で [動画をログ]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) することもできます。

Gymnasium と CleanRL ライブラリの併用方法の詳細は、この [report](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw) をご覧ください。 

{{< img src="/images/integrations/gymnasium.png" alt="Gymnasium の RL 環境" >}}