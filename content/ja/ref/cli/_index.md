---
title: コマンドライン インターフェース
menu:
  reference:
    identifier: ja-ref-cli-_index
---

**使用法**

`wandb [OPTIONS] COMMAND [ARGS]...`

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--version` | バージョンを表示して終了します。 |

**コマンド**

| **コマンド** | **説明** |
| :--- | :--- |
| agent | W&B エージェントを実行します |
| artifact | アーティファクトと対話するためのコマンド |
| beta | wandb CLI コマンドのベータバージョン。 |
| controller | W&B ローカルsweepコントローラを実行します |
| disabled | W&B を無効にします。 |
| docker | コードをdockerコンテナで実行します。 |
| docker-run | `docker run` をラップし、WANDB_API_KEY と WANDB_DOCKER を追加します... |
| enabled | W&B を有効にします。 |
| init | Weights & Biasesでディレクトリーを設定します |
| job | W&B ジョブを管理および表示するためのコマンド |
| launch | W&B Jobを起動またはキューに入れます。 |
| launch-agent | W&B ローンンチ エージェントを実行します。 |
| launch-sweep | W&B ローンンチ スウィープを実行します（実験的）。 |
| login | Weights & Biases にログインします |
| offline | W&B 同期を無効にします |
| online | W&B 同期を有効にします |
| pull | Weights & Biases からファイルを取得します |
| restore | runのコード、設定、およびdocker状態を復元します |
| scheduler | W&B ローンンチ スウィープ スケジューラを実行します（実験的） |
| server | ローカル W&B サーバーを操作するためのコマンド |
| status | 設定情報を表示します |
| sweep | ハイパーパラメーター探索を初期化します。 |
| sync | オフライントレーニングディレクトリーを W&B にアップロードします |
| verify | ローカルインスタンスを検証します |