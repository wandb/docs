---
title: Farama Gymnasium
description: W&B を Farama Gymnasium と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-farama-gymnasium
    parent: integrations
weight: 90
---

もし [Farama Gymnasium](https://gymnasium.farama.org/#) を使用している場合、`gymnasium.wrappers.Monitor` によって生成された環境のビデオを自動的にログします。ただし、[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) に `monitor_gym` キーワード引数を `True` に設定してください。

私たちの gymnasium のインテグレーションは非常に軽量です。単に `gymnasium` からログされるビデオファイルの[名前を確認](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67)し、それにちなんで名前を付けるか、一致するものが見つからない場合は `"videos"` を使用します。より多くの制御を希望する場合は、手動で[ビデオをログ](./guides/models/track/log/media.md)することもできます。

Gymnasium を CleanRL ライブラリと組み合わせて使用する方法について詳しく知りたい場合は、この[レポート](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw)をチェックしてください。

{{< img src="/images/integrations/gymnasium.png" alt="" >}}