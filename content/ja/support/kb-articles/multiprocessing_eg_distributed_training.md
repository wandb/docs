---
title: How can I use wandb with multiprocessing, e.g. distributed training?
menu:
  support:
    identifier: ja-support-kb-articles-multiprocessing_eg_distributed_training
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

もしトレーニングプログラムが複数のプロセスを使用する場合、`wandb.init()` なしでプロセスから wandb のメソッド呼び出しをしないようにプログラムを構成してください。

マルチプロセス のトレーニングは、次の方法で管理します。

1. 全てのプロセスで `wandb.init` を呼び出し、[group]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) キーワード 引数 を使用して、共有 グループ を作成します。各プロセスは独自の wandb run を持ち、UI はトレーニング プロセスをまとめてグループ化します。
2. 1つのプロセスからのみ `wandb.init` を呼び出し、[multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) を介して ログ に記録するデータを渡します。

{{% alert %}}
Torch DDP を使用した コード 例など、これらのアプローチの詳細な説明については、[分散トレーニング ガイド]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。
{{% /alert %}}
