---
title: 私のトレーニングプロセスで wandb.init は何をしますか？
menu:
  support:
    identifier: ja-support-kb-articles-wandbinit_training_process
support:
- 環境変数
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init()` がトレーニングスクリプトで実行されると、API 呼び出しによりサーバー上に run オブジェクトが作成されます。新しいプロセスが開始され、ストリームとメトリクスの収集が行われ、主要なプロセスは通常通り機能します。スクリプトはローカルファイルに書き込みを行い、別のプロセスがシステムメトリクスを含むデータをサーバーにストリームします。ストリーミングをオフにするには、トレーニングディレクトリから `wandb off` を実行するか、`WANDB_MODE` 環境変数を `offline` に設定します。