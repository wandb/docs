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

トレーニングプログラムが複数の プロセス を使用する場合、`wandb.init()` なしで プロセス から wandb の メソッド 呼び出しを行わないようにプログラムを構成してください。

以下の方法でマルチ プロセス トレーニング を管理します。

1. すべての プロセス で `wandb.init` を呼び出し、[group]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) キーワード 引数 を使用して共有 グループ を作成します。各 プロセス は独自の wandb run を持ち、UI は トレーニング プロセス をまとめて グループ 化します。
2. 1つの プロセス からのみ `wandb.init` を呼び出し、[multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) を通して ログ に記録する data を渡します。

{{% alert %}}
Torch DDP を使用した コード 例など、これらのアプローチの詳細な説明については、[分散 トレーニング ガイド]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。
{{% /alert %}}
