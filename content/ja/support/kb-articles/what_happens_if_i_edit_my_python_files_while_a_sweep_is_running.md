---
title: sweep の実行中に自分の Python ファイルを編集したらどうなりますか？
menu:
  support:
    identifier: ja-support-kb-articles-what_happens_if_i_edit_my_python_files_while_a_sweep_is_running
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

sweep 実行中は:
- sweep が使用する `train.py` スクリプトが変更されても、sweep は元の `train.py` を使い続けます
- `train.py` スクリプトが参照するファイル（たとえば `helper.py` スクリプト内のヘルパー関数）が変更された場合、sweep は更新された `helper.py` の使用を開始します。