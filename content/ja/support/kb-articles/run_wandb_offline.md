---
title: wandb をオフラインで実行できますか?
menu:
  support:
    identifier: ja-support-kb-articles-run_wandb_offline
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

トレーニング をオフライン マシンで行う場合、サーバー に 結果 をアップロードするには次の手順に従ってください:

1. インターネット接続なしでローカルに メトリクス を保存するには、環境 変数 `WANDB_MODE=offline` を設定します。
2. アップロードの準備ができたら、ディレクトリー で `wandb init` を実行して Project 名を設定します。
3. クラウド サービスへ メトリクス を転送し、ホスト型 Web アプリで 結果 に アクセス するには、`wandb sync YOUR_RUN_DIRECTORY` を使用します。

run がオフラインであることを確認するには、`wandb.init()` 実行後に `run.settings._offline` または `run.settings.mode` を確認してください。