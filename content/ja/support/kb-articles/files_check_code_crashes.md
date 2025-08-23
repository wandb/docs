---
title: コードがクラッシュしたときに確認すべきファイルはどれですか？
menu:
  support:
    identifier: ja-support-kb-articles-files_check_code_crashes
support:
- ログ
toc_hide: true
type: docs
url: /support/:filename
---

該当する run については、コードが実行されているディレクトリー内の `wandb/run-<date>_<time>-<run-id>/logs` にある `debug.log` および `debug-internal.log` を確認してください。