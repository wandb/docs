---
title: How can I use wandb with multiprocessing, e.g. distributed training?
menu:
  support:
    identifier: ja-support-multiprocessing_eg_distributed_training
tags:
- experiments
toc_hide: true
type: docs
---

トレーニングプログラムが複数のプロセスを使用する場合、`wandb.init()`なしでプロセスが wandb メソッドを呼び出さないようにプログラムを構成してください。

マルチプロセストレーニングを管理するためのアプローチは以下の通りです：

1. すべてのプロセスで `wandb.init` を呼び出し、[group]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) キーワード引数を使用して共有グループを作成します。各プロセスは独自の wandb run を持ち、UI はトレーニングプロセスをまとめてグループ化します。
2. 1つのプロセスでのみ `wandb.init` を呼び出し、[multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) を通じてログにデータを渡します。

{{% alert %}}
これらのアプローチに関する詳細な説明や、Torch DDP のコード例については、[Distributed Training Guide]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。
{{% /alert %}}