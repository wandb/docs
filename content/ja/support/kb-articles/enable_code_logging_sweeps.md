---
title: スイープでコードログを有効にするにはどうすれば良いですか？
menu:
  support:
    identifier: ja-support-kb-articles-enable_code_logging_sweeps
support:
- スイープ
toc_hide: true
type: docs
url: /ja/support/:filename
---

スイープのためのコード ログを有効にするには、W&B Run を初期化した後に `wandb.log_code()` を追加します。この操作は、W&B のプロファイル設定でコード ログが有効になっている場合でも必要です。詳細なコード ログについては、[こちらの `wandb.log_code()` ドキュメント]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}})を参照してください。