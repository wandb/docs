---
title: '`wandb` ファイルのローカル場所をどのように定義できますか?'
menu:
  support:
    identifier: >-
      ja-support-kb-articles-how_can_i_define_the_local_folder_where_to_save_the_wandb_files
support:
  - environment variables
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: how_can_i_define_the_local_folder_where_to_save_the_wandb_files
---
- `WANDB_DIR=<path>` または `wandb.init(dir=<path>)`: トレーニングスクリプトのために作成された `wandb` フォルダの場所を制御します。デフォルトは `./wandb` です。このフォルダには Run のデータとログが保存されます。
- `WANDB_ARTIFACT_DIR=<path>` または `wandb.Artifact().download(root="<path>")`: アーティファクトがダウンロードされる場所を制御します。デフォルトは `./artifacts` です。
- `WANDB_CACHE_DIR=<path>`: これは `wandb.Artifact` を呼び出した際にアーティファクトが作成・保存される場所です。デフォルトは `~/.cache/wandb` です。
- `WANDB_CONFIG_DIR=<path>`: 設定ファイルが保存される場所です。デフォルトは `~/.config/wandb` です。
- `WANDB_DATA_DIR=<PATH>`: アップロード中のアーティファクトのステージングに使用される場所を制御します。デフォルトは `~/.cache/wandb-data/` です。