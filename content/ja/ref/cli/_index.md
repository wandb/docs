---
title: Command Line Interface
menu:
  reference:
    identifier: ja-ref-cli-_index
---

**使用方法**

`wandb [OPTIONS] COMMAND [ARGS]...`

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--version` | バージョンを表示して終了します。 |

**コマンド**

| **コマンド** | **説明** |
| :--- | :--- |
| agent | W&B エージェント を実行します。 |
| artifact | アーティファクト を操作するための コマンド です。 |
| beta | wandb CLI コマンド のベータ版です。 |
| controller | W&B ローカル sweep コントローラ を実行します。 |
| disabled | W&B を無効にします。 |
| docker | dockerコンテナ でコードを実行します。 |
| docker-run | `docker run` をラップし、WANDB_API_KEY と WANDB_DOCKER... を追加します。 |
| enabled | W&B を有効にします。 |
| init | Weights & Biases で ディレクトリー を設定します。 |
| job | W&B ジョブ を管理および表示するための コマンド です。 |
| launch | W&B ジョブ を ローンチ または キュー に入れます。 |
| launch-agent | W&B ローンチ エージェント を実行します。 |
| launch-sweep | W&B ローンチ sweep を実行します（試験的）。 |
| login | Weights & Biases にログインします。 |
| offline | W&B の同期を無効にします。 |
| online | W&B の同期を有効にします。 |
| pull | Weights & Biases から ファイル をプルします。 |
| restore | run の コード 、設定、および docker の状態を復元します。 |
| scheduler | W&B ローンチ sweep スケジューラ を実行します（試験的）。 |
| server | ローカル W&B サーバー を操作するための コマンド です。 |
| status | 設定 を表示します。 |
| sweep | ハイパーパラメーター探索 を初期化します。 |
| sync | オフライン の トレーニング ディレクトリー を W&B にアップロードします。 |
| verify | ローカル インスタンス を検証します。 |
