---
title: 分散トレーニングのようなマルチプロセッシングで wandb をどのように使用できますか？
menu:
  support:
    identifier: ja-support-kb-articles-multiprocessing_eg_distributed_training
support:
- 実験管理
toc_hide: true
type: docs
url: /ja/support/:filename
---

トレーニングプログラムが複数のプロセスを使用する場合、`wandb.init()` なしでプロセスから wandb メソッド呼び出しを行わないようにプログラムを構築してください。

マルチプロセスのトレーニングを管理するには、以下のアプローチを使用します。

1. すべてのプロセスで `wandb.init` を呼び出し、[group]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) キーワード引数を使用して共有グループを作成します。各プロセスは独自の wandb run を持ち、UI はトレーニング プロセスを一緒にグループ化します。
2. ただ一つのプロセスから `wandb.init` を呼び出し、[multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) を通じてログ記録するデータを渡します。

{{% alert %}}
これらのアプローチの詳細な説明と、Torch DDP のコード例を含む [分散トレーニングガイド]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。
{{% /alert %}}