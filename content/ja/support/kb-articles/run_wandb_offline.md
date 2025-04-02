---
title: Can I run wandb offline?
menu:
  support:
    identifier: ja-support-kb-articles-run_wandb_offline
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

オフラインマシンでトレーニングを行う場合は、次の手順で結果をサーバーにアップロードします。

1. 環境変数 `WANDB_MODE=offline` を設定して、インターネット接続なしでメトリクスをローカルに保存します。
2. アップロードの準備ができたら、ディレクトリーで `wandb init` を実行して、Project 名を設定します。
3. `wandb sync YOUR_RUN_DIRECTORY` を使用して、メトリクスをクラウドサービスに転送し、ホストされているウェブアプリで結果にアクセスします。

run がオフラインであることを確認するには、`wandb.init()` の実行後に `run.settings._offline` または `run.settings.mode` を確認します。