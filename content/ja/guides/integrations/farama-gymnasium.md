---
title: Farama Gymnasium
description: W&B と Farama Gymnasium を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-farama-gymnasium
    parent: integrations
weight: 90
---

[Farama Gymnasium](https://gymnasium.farama.org/#) を使用している場合、`gymnasium.wrappers.Monitor` で生成された環境の動画は自動的にログに記録されます。`monitor_gym` キーワード引数を [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) に `True` で設定してください。

Gymnasium のインテグレーション は非常に軽量です。動画ファイルの名前を見て、その名前にちなんで名前を付けます。一致するものが見つからない場合は、`"videos"` にフォールバックします。詳細な制御が必要な場合は、いつでも手動で [動画をログに記録]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) できます。 ([動画ファイル名の確認](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67))

CleanRL ライブラリで Gymnasium を使用する方法については、[レポート](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw) をご覧ください。

{{< img src="/images/integrations/gymnasium.png" alt="" >}}
