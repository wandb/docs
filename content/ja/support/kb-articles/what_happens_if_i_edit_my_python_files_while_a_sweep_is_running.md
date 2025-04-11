---
title: Python ファイルを編集すると、sweep 実行中に何が起こりますか？
menu:
  support:
    identifier: ja-support-kb-articles-what_happens_if_i_edit_my_python_files_while_a_sweep_is_running
support:
- sweep
toc_hide: true
type: docs
url: /support/:filename
---

スイープが実行中の場合:
- スイープが使用する `train.py` スクリプトが変更された場合、スイープは元の `train.py` を使用し続けます。
- スイープが参照する `train.py` スクリプト内のファイルが変更された場合、例えば `helper.py` スクリプト内の補助関数など、スイープは更新された `helper.py` を使用し始めます。