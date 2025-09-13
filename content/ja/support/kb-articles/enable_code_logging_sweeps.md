---
title: Sweeps で コードのログを有効にするには？
menu:
  support:
    identifier: ja-support-kb-articles-enable_code_logging_sweeps
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

Sweeps でコード ログを有効にするには、W&B Run を初期化した後に `wandb.log_code()` を追加します。これは、W&B プロファイル設定でコード ログが有効になっている場合でも必要です。より高度なコード ログについては、[こちらの `wandb.log_code()` のドキュメント]({{< relref path="/ref/python/sdk/classes/run#log_code" lang="ja" >}})を参照してください。