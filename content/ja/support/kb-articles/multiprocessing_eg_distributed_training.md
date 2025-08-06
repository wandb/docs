---
title: wandb をマルチプロセッシング、例えば分散トレーニングで使うにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-multiprocessing_eg_distributed_training
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

トレーニングプログラムが複数のプロセスを使用する場合は、`wandb.init()` を呼び出していないプロセスから wandb メソッドを呼び出さないように、プログラムを構成してください。

マルチプロセスのトレーニングは、以下の方法で管理できます。

1. すべてのプロセスで `wandb.init` を呼び出し、[group]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) キーワード引数を使って共通のグループを作成します。各プロセスで個別の wandb run が作成され、UI 上ではトレーニングプロセスがまとめて表示されます。
2. 1つのプロセスのみで `wandb.init` を呼び出し、[multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) 経由でログに記録するデータを渡します。

{{% alert %}}
これらのアプローチの詳細や Torch DDP を用いたコード例については、[分散トレーニングガイド]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。
{{% /alert %}}