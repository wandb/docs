---
title: wandb を マルチプロセッシング（例えば分散トレーニング）で使うにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-multiprocessing_eg_distributed_training
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング プログラムが複数のプロセスを使用する場合、`wandb.init()` を実行していないプロセスから wandb の メソッド呼び出しを行わないように、プログラムを構成してください。

マルチプロセスのトレーニングは次のアプローチで管理します:

1. すべてのプロセスで `wandb.init` を呼び出し、共有グループを作成するために [グループ]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) キーワード引数を使用します。各プロセスは自身の wandb run を持ち、UI はトレーニング プロセスをまとめて表示します。
2. 1 つのプロセスからのみ `wandb.init` を呼び出し、[multiprocessing のキュー](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) を介して、ログに記録するデータを渡します.

{{% alert %}}
これらのアプローチの詳細な解説 (Torch DDP の コード例を含む) については、[分散トレーニング ガイド]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。
{{% /alert %}}