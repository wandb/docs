---
title: How can I define the local location for `wandb` files?
menu:
  support:
    identifier: ja-support-how_can_i_define_the_local_folder_where_to_save_the_wandb_files
tags:
- environment variables
- experiments
toc_hide: true
type: docs
---

- `WANDB_DIR=<path>` または `wandb.init(dir=<path>)`: トレーニングスクリプトのために作成された `wandb` フォルダの場所を制御します。デフォルトは `./wandb` です。このフォルダには Run のデータとログが保存されます。
- `WANDB_ARTIFACT_DIR=<path>` または `wandb.Artifact().download(root="<path>")`: アーティファクトがダウンロードされる場所を制御します。デフォルトは `artifacts/` です。
- `WANDB_CACHE_DIR=<path>`: `wandb.Artifact` を呼び出したときにアーティファクトが作成および保存される場所です。デフォルトは `~/.cache/wandb` です。
- `WANDB_CONFIG_DIR=<path>`: 設定ファイルが保存される場所です。デフォルトは `~/.config/wandb` です。
- `WANDB_DATA_DIR=<PATH>`: アップロード中にアーティファクトをステージングするために使用される場所です。デフォルトはプラットフォームによって異なり、`platformdirs` Python パッケージの `user_data_dir` を使用します。