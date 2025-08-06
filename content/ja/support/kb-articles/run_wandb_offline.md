---
title: wandb をオフラインで使えますか？
menu:
  support:
    identifier: ja-support-kb-articles-run_wandb_offline
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

トレーニングがオフラインマシンで実行される場合、以下の手順で結果をサーバーにアップロードできます。

1. 環境変数 `WANDB_MODE=offline` を設定して、インターネット接続なしでメトリクスをローカルに保存します。
2. アップロードの準備ができたら、ディレクトリーで `wandb init` を実行して Project 名を設定します。
3. `wandb sync YOUR_RUN_DIRECTORY` を使ってメトリクスをクラウドサービスへ転送し、ホストされた Web アプリで結果にアクセスします。

run がオフラインであることを確認するには、`wandb.init()` 実行後に `run.settings._offline` または `run.settings.mode` をチェックしてください。