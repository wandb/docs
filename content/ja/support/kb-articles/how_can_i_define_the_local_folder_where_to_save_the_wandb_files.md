---
title: '`wandb` ファイルのローカル保存先はどのように指定できますか？'
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_define_the_local_folder_where_to_save_the_wandb_files
support:
- 環境変数
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

- `WANDB_DIR=<path>` または `wandb.init(dir=<path>)`: トレーニングスクリプト用に作成される `wandb` フォルダの保存先を指定します。デフォルトは `./wandb` です。このフォルダには Run の データ と ログ が保存されます
- `WANDB_ARTIFACT_DIR=<path>` または `wandb.Artifact().download(root="<path>")`: Artifacts がダウンロードされる場所を指定します。デフォルトは `./artifacts` です
- `WANDB_CACHE_DIR=<path>`: これは、`wandb.Artifact` を呼び出したときに Artifacts が作成・保存される場所です。デフォルトは `~/.cache/wandb` です
- `WANDB_CONFIG_DIR=<path>`: 設定ファイルが保存される場所です。デフォルトは `~/.config/wandb` です
- `WANDB_DATA_DIR=<PATH>`: アップロード中に Artifacts の一時保存先となる場所を指定します。デフォルトは `~/.cache/wandb-data/` です