---
title: wandb をマルチプロセッシング、例えば分散トレーニングで使うにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

トレーニングプログラムが複数のプロセスを使用する場合、`wandb.init()` が実行されていないプロセスから wandb メソッド呼び出しを行わないように、プログラム構成を工夫してください。

マルチプロセスでのトレーニング管理には、以下のアプローチを利用できます。

1. すべてのプロセスで `wandb.init` を呼び出し、[group]({{< relref "/guides/models/track/runs/grouping.md" >}}) キーワード引数を使って共通のグループを作成します。それぞれのプロセスは独自の wandb run を持ち、UI 上でトレーニングプロセスがまとめて表示されます。
2. 1つのプロセスだけで `wandb.init` を呼び出し、[multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) を使ってログに渡したいデータをやり取りします。

{{% alert %}}
これらのアプローチについて詳しくは、コード例（Torch DDP を含む）とあわせて [Distributed Training Guide]({{< relref "/guides/models/track/log/distributed-training.md" >}}) をご参照ください。
{{% /alert %}}