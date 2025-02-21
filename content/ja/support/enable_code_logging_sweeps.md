---
title: How do I enable code logging with Sweeps?
menu:
  support:
    identifier: ja-support-enable_code_logging_sweeps
tags:
- sweeps
toc_hide: true
type: docs
---

Sweeps でのコード ログを有効にするには、W&B Run の初期化後に `wandb.log_code()` を追加します。W&B プロファイルの設定でコード ログが有効になっている場合でも、この操作は必要です。高度なコード ログについては、[`wandb.log_code()` のドキュメントはこちら]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}})を参照してください。
