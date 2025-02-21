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

sweep が実行中の場合:
- sweep が使用する `train.py` スクリプトが変更された場合でも、sweep は元の `train.py` を使用し続けます
- `train.py` スクリプトが参照するファイルが変更された場合、例えば `helper.py` スクリプトのヘルパー関数などが変更されると、sweep は更新された `helper.py` を使用し始めます。