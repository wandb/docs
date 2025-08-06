---
title: コマンドラインインターフェース
---

**使用方法**

`wandb [OPTIONS] COMMAND [ARGS]...`



**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--version` | バージョン情報を表示して終了します。 |


**コマンド**

| **コマンド** | **説明** |
| :--- | :--- |
| agent | W&B エージェントを実行します |
| artifact | Artifacts を操作するためのコマンド |
| beta | wandb CLI コマンドのベータバージョンです。 |
| controller | W&B のローカルsweepコントローラを実行します |
| disabled | W&B を無効化します。 |
| docker | コードを dockerコンテナ 内で実行します。 |
| docker-run | `docker run` をラップし、WANDB_API_KEYとWANDB_DOCKER... を追加します |
| enabled | W&B を有効化します。 |
| init | ディレクトリーを Weights & Biases で設定します |
| job | W&B Jobの管理や表示のためのコマンド |
| launch | W&B Jobを実行またはキューへ追加します。 |
| launch-agent | W&B launch エージェントを実行します。 |
| launch-sweep | W&B launch sweep を実行します（実験的機能）。 |
| login | Weights & Biases にログインします |
| offline | W&Bの同期を無効化します |
| online | W&Bの同期を有効化します |
| pull | Weights & Biases からファイルを取得します |
| restore | runのコード、設定、docker状態を復元します |
| scheduler | W&B launch sweep scheduler を実行します（実験的機能） |
| server | ローカル W&B サーバーを操作するためのコマンド |
| status | 設定内容を表示します |
| sweep | ハイパーパラメーターsweepを初期化します。 |
| sync | オフライントレーニングディレクトリーを W&B にアップロードします |
| verify | ローカルインスタンスを検証します |