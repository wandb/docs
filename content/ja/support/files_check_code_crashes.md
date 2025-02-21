---
title: Which files should I check when my code crashes?
menu:
  support:
    identifier: ja-support-files_check_code_crashes
tags:
- logs
toc_hide: true
type: docs
---

影響を受けた run については、コードが実行されているディレクトリーの `wandb/run-<date>_<time>-<run-id>/logs` にある `debug.log` と `debug-internal.log` を確認してください。