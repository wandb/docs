---
title: What happens if I edit my Python files while a sweep is running?
menu:
  support:
    identifier: ja-support-kb-articles-what_happens_if_i_edit_my_python_files_while_a_sweep_is_running
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

sweep の実行中:
- sweep で使用される `train.py` スクリプトが変更された場合、sweep は引き続き元の `train.py` を使用します。
- `train.py` スクリプトが参照するファイル（`helper.py` スクリプト内のヘルパー関数など）が変更された場合、sweep は更新された `helper.py` を使用し始めます。