---
title: sweep が実行中に Python ファイルを編集した場合、どうなりますか？
menu:
  support:
    identifier: ja-support-kb-articles-what_happens_if_i_edit_my_python_files_while_a_sweep_is_running
support:
- スイープ
toc_hide: true
type: docs
url: /support/:filename
---

sweep 実行中の挙動について:
- sweep で使用している `train.py` スクリプトが変更された場合でも、sweep は元の `train.py` を引き続き使用します
- `train.py` スクリプトから参照されているファイル（例えば、`helper.py` スクリプト内のヘルパー関数など）が変更された場合、sweep は更新後の `helper.py` を使用し始めます