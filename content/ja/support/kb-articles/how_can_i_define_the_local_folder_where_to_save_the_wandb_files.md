---
title: '`wandb` ファイルのローカル保存先はどのように指定できますか？'
url: /support/:filename
toc_hide: true
type: docs
support:
- 環境変数
- 実験
---

- `WANDB_DIR=<path>` または `wandb.init(dir=<path>)`: トレーニングスクリプト用に作成される `wandb` フォルダの保存先を指定します。デフォルトは `./wandb` です。このフォルダには Run のデータやログが保存されます。
- `WANDB_ARTIFACT_DIR=<path>` または `wandb.Artifact().download(root="<path>")`: Artifacts のダウンロード先を指定します。デフォルトは `./artifacts` です。
- `WANDB_CACHE_DIR=<path>`: `wandb.Artifact` を呼び出した時に Artifacts が作成・保存される場所です。デフォルトは `~/.cache/wandb` です。
- `WANDB_CONFIG_DIR=<path>`: 設定ファイルの保存先です。デフォルトは `~/.config/wandb` です。
- `WANDB_DATA_DIR=<PATH>`: アップロード時に Artifacts の一時保存に使われる場所を指定します。デフォルトは `~/.cache/wandb-data/` です。