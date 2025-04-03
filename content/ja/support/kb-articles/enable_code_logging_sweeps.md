---
title: How do I enable code logging with Sweeps?
menu:
  support:
    identifier: ja-support-kb-articles-enable_code_logging_sweeps
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

Sweeps のコード ログ記録を有効にするには、W&B Run の初期化後に `wandb.log_code()` を追加します。この操作は、W&B プロファイル 設定 でコード ログ記録が有効になっている場合でも必要です。高度なコード ログ記録については、[ `wandb.log_code()` のドキュメントはこちら]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}}) を参照してください。
