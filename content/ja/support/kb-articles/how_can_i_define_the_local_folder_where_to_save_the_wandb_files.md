---
title: How can I define the local location for `wandb` files?
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_define_the_local_folder_where_to_save_the_wandb_files
support:
- environment variables
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

- `WANDB_DIR=<path>` または `wandb.init(dir=<path>)`: トレーニングスクリプト 用に作成された `wandb` フォルダーの場所を制御します。デフォルトは `./wandb` です。このフォルダーには、 Run のデータと ログ が保存されます。
- `WANDB_ARTIFACT_DIR=<path>` または `wandb.Artifact().download(root="<path>")`: Artifacts がダウンロードされる場所を制御します。デフォルトは `./artifacts` です。
- `WANDB_CACHE_DIR=<path>`: これは、 `wandb.Artifact` を呼び出すときに Artifacts が作成および保存される場所です。デフォルトは `~/.cache/wandb` です。
- `WANDB_CONFIG_DIR=<path>`: 構成ファイルが保存される場所。デフォルトは `~/.config/wandb` です。
- `WANDB_DATA_DIR=<PATH>`: アップロード中に Artifacts のステージングに使用される場所を制御します。デフォルトは `~/.cache/wandb-data/` です。
