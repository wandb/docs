---
title: wandb をオフラインで実行することはできますか?
menu:
  support:
    identifier: ja-support-kb-articles-run_wandb_offline
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: run_wandb_offline
---
オフラインマシンでトレーニングが行われる場合、以下の手順を使用して結果をサーバーにアップロードします:

1. 環境変数 `WANDB_MODE=offline` を設定し、インターネット接続なしでメトリクスをローカルに保存します。
2. アップロードの準備ができたら、ディレクトリーで `wandb init` を実行してプロジェクト名を設定します。
3. `wandb sync YOUR_RUN_DIRECTORY` を使用してメトリクスをクラウドサービスに転送し、ホストされたウェブアプリで結果にアクセスします。

Run がオフラインであることを確認するには、`wandb.init()` を実行した後に `run.settings._offline` または `run.settings.mode` を確認してください。