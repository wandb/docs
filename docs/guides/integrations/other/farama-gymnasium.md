---
description: W&B を Farama Gymnasium と統合する方法
slug: /guides/integrations/farama-gymnasium
displayed_sidebar: default
---


# Farama Gymnasium

もし [Farama Gymnasium](https://gymnasium.farama.org/#) を使用している場合、`gymnasium.wrappers.Monitor` によって生成された環境のビデオを自動的にログします。`monitor_gym` キーワード引数を [`wandb.init`](../../../ref/python/init.md) に `True` と設定するだけです。

私たちの gymnasium インテグレーションは非常に軽量です。`gymnasium` からログされる [ビデオファイルの名前を確認し](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67)、それに従って名前を付けるか、見つからなかった場合は `"videos"` にフォールバックします。より詳細に制御したい場合は、手動で [動画をログする](../../track/log/media.md) こともできます。

Gymnasium と CleanRL ライブラリの使用方法については、この [レポート](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw) をチェックしてください。

![](/images/integrations/gymnasium.png)