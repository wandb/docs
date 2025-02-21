---
title: What happens if I edit my Python files while a sweep is running?
menu:
  support:
    identifier: ja-support-what_happens_if_i_edit_my_python_files_while_a_sweep_is_running
tags:
- sweeps
toc_hide: true
type: docs
---

sweep の実行中:
- sweep が使用する `train.py` スクリプトが変更された場合、sweep は引き続き元の `train.py` を使用します。
- `train.py` スクリプトが参照するファイル (たとえば、`helper.py` スクリプト内のヘルパー関数など) が変更された場合、sweep は更新された `helper.py` の使用を開始します。
