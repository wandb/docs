---
title: Farama Gymnasium
description: Farama Gymnasium と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-farama-gymnasium
    parent: integrations
weight: 90
---

[Farama Gymnasium](https://gymnasium.farama.org/#) を使用している場合、`gymnasium.wrappers.Monitor` で生成された環境の動画が自動的にログに記録されます。`monitor_gym` キーワード 引数 を [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) に `True` に設定するだけです。

当社の Gymnasium インテグレーション は非常に軽量です。 `gymnasium` からログに記録されている [動画ファイルの名前を確認](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67) し、その名前を付けます。一致するものが見つからない場合は、`"videos"` にフォールバックします。より詳細な制御が必要な場合は、いつでも手動で [動画をログに記録]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) できます。

CleanRL ライブラリ で Gymnasium を使用する方法の詳細については、こちらの [Reports](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw) をご覧ください。

{{< img src="/images/integrations/gymnasium.png" alt="" >}}
