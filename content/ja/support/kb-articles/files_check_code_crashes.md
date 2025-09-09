---
title: コードがクラッシュしたとき、どのファイルを確認すればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-files_check_code_crashes
support:
- ログ
toc_hide: true
type: docs
url: /support/:filename
---

対象の run については、コードを実行しているディレクトリーの `wandb/run-<date>_<time>-<run-id>/logs` にある `debug.log` と `debug-internal.log` を確認してください。