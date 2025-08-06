---
title: コードがクラッシュしたとき、どのファイルを確認すればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- ログ
---

影響を受けた run については、コードが実行されているディレクトリー内の `wandb/run-<date>_<time>-<run-id>/logs` にある `debug.log` と `debug-internal.log` を確認してください。