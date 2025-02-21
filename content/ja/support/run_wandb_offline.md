---
title: Can I run wandb offline?
menu:
  support:
    identifier: ja-support-run_wandb_offline
tags:
- experiments
toc_hide: true
type: docs
---

オフラインのマシンでトレーニングを行う場合、結果をサーバーにアップロードするには次のステップを使用します:

1. 環境変数 `WANDB_MODE=offline` を設定し、インターネット接続なしでメトリクスをローカルに保存します。
2. アップロードの準備ができたら、ディレクトリーで `wandb init` を実行し、プロジェクト名を設定します。
3. `wandb sync YOUR_RUN_DIRECTORY` を使用して、メトリクスをクラウドサービスに転送し、ホストされたウェブアプリで結果にアクセスします。

run がオフラインであることを確認するには、`wandb.init()` を実行した後で `run.settings._offline` または `run.settings.mode` をチェックしてください。