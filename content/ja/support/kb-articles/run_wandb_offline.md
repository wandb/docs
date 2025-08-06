---
title: Wandb をオフラインで実行できますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

オフラインマシンでトレーニングを行う場合、結果をサーバーにアップロードするには次の手順を実行してください。

1. 環境変数 `WANDB_MODE=offline` を設定して、インターネット接続なしでメトリクスをローカルに保存します。
2. アップロードの準備ができたら、ディレクトリー内で `wandb init` を実行し、Project 名を設定します。
3. `wandb sync YOUR_RUN_DIRECTORY` を使ってメトリクスをクラウドサービスに転送し、ホストされたウェブアプリで結果にアクセスします。

run がオフラインモードであることを確認するには、`wandb.init()` 実行後に `run.settings._offline` または `run.settings.mode` をチェックしてください。