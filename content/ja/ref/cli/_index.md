---
title: コマンドラインインターフェース
menu:
  reference:
    identifier: ja-ref-cli-_index
---

**使い方**

`wandb [OPTIONS] COMMAND [ARGS]...`



**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--version` | バージョン情報を表示して終了します。 |


**コマンド**

| **コマンド** | **説明** |
| :--- | :--- |
| agent | W&B エージェントを実行します |
| artifact | Artifacts とやり取りするためのコマンドです |
| beta | wandb CLI コマンドのベータバージョンです。 |
| controller | W&B ローカル sweep コントローラを実行します |
| disabled | W&B を無効化します。 |
| docker | コードを docker コンテナで実行します。 |
| docker-run | `docker run` をラップし、WANDB_API_KEY と WANDB_DOCKER... を追加します |
| enabled | W&B を有効化します。 |
| init | ディレクトリーを Weights & Biases 用に設定します |
| job | W&B ジョブの管理と表示コマンド |
| launch | W&B Job を起動またはキューに追加します。|
| launch-agent | W&B Launch エージェントを実行します。|
| launch-sweep | W&B Launch Sweep（実験的）を実行します。|
| login | Weights & Biases にログインします |
| offline | W&B の同期を無効化します |
| online | W&B の同期を有効化します |
| pull | Weights & Biases からファイルを取得します |
| restore | run のコード、設定、docker 状態を復元します |
| scheduler | W&B Launch Sweep スケジューラ（実験的）を実行します |
| server | ローカル W&B サーバー操作用コマンド |
| status | 設定内容を表示します |
| sweep | ハイパーパラメーター sweep を初期化します。 |
| sync | オフラインのトレーニング ディレクトリーを W&B にアップロードします |
| verify | ローカルインスタンスを検証します |