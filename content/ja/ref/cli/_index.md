---
title: コマンドライン インターフェース
menu:
  reference:
    identifier: ja-ref-cli-_index
weight: 2
---

**使い方**

`wandb [OPTIONS] COMMAND [ARGS]...`



**オプション**

| **Option** | **Description** |
| :--- | :--- |
| `--version` | バージョンを表示して終了します。 |


**コマンド**

| **Command** | **Description** |
| :--- | :--- |
| agent | W&B エージェントを実行します |
| artifact | artifacts とやり取りするためのコマンド |
| beta | wandb CLI コマンドのベータ版。 |
| controller | W&B のローカル sweep コントローラを実行します |
| disabled | W&B を無効化します。 |
| docker | docker コンテナでコードを実行します。 |
| docker-run | `docker run` をラップし、WANDB_API_KEY と WANDB_DOCKER... を追加します。 |
| enabled | W&B を有効化します。 |
| init | Weights & Biases 用にディレクトリーを設定します |
| job | W&B ジョブの管理と表示のためのコマンド |
| launch | W&B Job を Launch する、またはキューに入れます。 |
| launch-agent | W&B Launch エージェントを実行します。 |
| launch-sweep | W&B Launch sweep を実行します（実験的）。 |
| login | Weights & Biases にログインします |
| offline | W&B の同期を無効化します |
| online | W&B の同期を有効化します |
| pull | Weights & Biases からファイルを取得します |
| restore | run のためのコード、設定、docker の状態を復元します |
| scheduler | W&B Launch sweep スケジューラを実行します（実験的） |
| server | ローカルの W&B サーバーを操作するためのコマンド |
| status | 設定を表示します |
| sweep | ハイパーパラメーター sweep を初期化します。 |
| sync | オフラインのトレーニングディレクトリーを W&B にアップロードします |
| verify | ローカル インスタンスを検証します |