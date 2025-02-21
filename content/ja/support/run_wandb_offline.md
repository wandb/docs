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

オフラインマシンでトレーニング が行われる場合は、次の手順に従って 結果 を サーバー にアップロードします。

1. 環境 変数 `WANDB_MODE=offline` を設定して、インターネット接続なしで メトリクス をローカルに保存します。
2. アップロードの準備ができたら、 ディレクトリー で `wandb init` を実行して、 project 名を設定します。
3. `wandb sync YOUR_RUN_DIRECTORY` を使用して、 メトリクス を クラウド サービスに転送し、ホストされているウェブアプリで 結果 に アクセスします。

run がオフラインであることを確認するには、`wandb.init()` を実行した後で、`run.settings._offline` または `run.settings.mode` を確認してください。
