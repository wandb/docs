---
title: sweep 実行中に Python ファイルを編集した場合、どうなりますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- Sweeps
---

sweep 実行中は、以下のようになります。
- sweep で使用している `train.py` スクリプトが変更されても、sweep は元の `train.py` を使い続けます
- `train.py` スクリプトが参照しているファイル（例えば `helper.py` スクリプト内のヘルパー関数など）が変更された場合、sweep は更新された `helper.py` を使うようになります