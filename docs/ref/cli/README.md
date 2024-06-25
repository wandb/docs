
# Command Line Interface

**Usage**

`wandb [OPTIONS] COMMAND [ARGS]...`

**Options**

| **Option** | **Description** |
| :--- | :--- |
| --version | バージョンを表示して終了します。 |

**Commands**

| **Command** | **Description** |
| :--- | :--- |
| agent | W&B エージェントを実行します |
| artifact | Artifacts と対話するためのコマンド |
| beta | wandb CLI コマンドのベータ版。 |
| controller | W&B ローカル sweep コントローラを実行します |
| disabled | W&B を無効化します。 |
| docker | dockerコンテナでコードを実行します。 |
| docker-run | `docker run` をラップして WANDB_API_KEY と WANDB_DOCKER を追加します... |
| enabled | W&B を有効にします。 |
| init | Weights & Biases でディレクトリーを設定します |
| job | W&B ジョブを管理および表示するためのコマンド |
| launch | W&B ジョブを開始またはキューします。 |
| launch-agent | W&B launch エージェントを実行します。 |
| launch-sweep | W&B launch sweep を実行します（実験的）。 |
| login | Weights & Biases にログインします |
| offline | W&B の同期を無効にします |
| online | W&B の同期を有効にします |
| pull | Weights & Biases からファイルをプルします |
| restore | run のコード、config、および docker 状態を復元します |
| scheduler | W&B launch sweep スケジューラーを実行します（実験的） |
| server | ローカル W&B サーバーを操作するためのコマンド |
| status | 設定を表示します |
| sweep | ハイパーパラメーター探索を初期化します。 |
| sync | オフライントレーニングディレクトリーを W&B にアップロードします |
| verify | ローカルインスタンスを検証します |