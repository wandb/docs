---
title: スイープでコードのログを有効にするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-enable_code_logging_sweeps
support:
- スイープ
toc_hide: true
type: docs
url: /support/:filename
---

スイープでコードのログを有効にするには、W&B Run を初期化した後に `wandb.log_code()` を追加してください。これは、W&B のプロフィール設定でコードログが有効になっている場合でも必要なアクションです。高度なコードログについては、[こちらの `wandb.log_code()` のドキュメント]({{< relref path="/ref/python/sdk/classes/run#log_code" lang="ja" >}})をご参照ください。