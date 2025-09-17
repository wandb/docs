---
title: '`wandb` ファイルのローカルの保存先はどのように設定できますか？'
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

- `WANDB_DIR=<path>` または `wandb.init(dir=<path>)`: トレーニングスクリプトによって作成される `wandb` フォルダの場所を制御します。デフォルトは `./wandb` です。このフォルダには Run のデータとログが保存されます。
- `WANDB_ARTIFACT_DIR=<path>` または `wandb.Artifact().download(root="<path>")`: artifacts がダウンロードされる場所を制御します。デフォルトは `./artifacts` です。
- `WANDB_CACHE_DIR=<path>`: これは `wandb.Artifact` を呼び出したときに artifacts が作成・保存される場所です。デフォルトは `~/.cache/wandb` です。
- `WANDB_CONFIG_DIR=<path>`: 設定ファイルが保存される場所です。デフォルトは `~/.config/wandb` です。
- `WANDB_DATA_DIR=<PATH>`: アップロード中に artifacts をステージングするために使われる場所を制御します。デフォルトは `~/.cache/wandb-data/` です。