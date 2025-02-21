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

スイープのためのコードログを有効にするには、W&B の Run を初期化した後に `wandb.log_code()` を追加します。この操作は、W&B プロファイル設定でコードログが有効になっている場合でも必要です。高度なコードログについては、[こちらの `wandb.log_code()` のドキュメント]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}})を参照してください。