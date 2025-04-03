---
title: What does wandb.init do to my training process?
menu:
  support:
    identifier: ja-support-kb-articles-wandbinit_training_process
support:
- environment variables
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init()` がトレーニング スクリプトで実行されると、API 呼び出しによってサーバー上に run オブジェクトが作成されます。新しいプロセスが開始され、メトリクスがストリーミングおよび収集されるため、プライマリ プロセスは正常に機能します。スクリプトはローカル ファイルに書き込みますが、別のプロセスがシステム メトリクスを含むデータをサーバーにストリーミングします。ストリーミングをオフにするには、トレーニング ディレクトリーから `wandb off` を実行するか、`WANDB_MODE` 環境変数を `offline` に設定します。
