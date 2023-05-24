# コマンドラインインターフェース

**使い方**

`wandb [オプション] コマンド [ARG]...`

**オプション**
| **オプション** | **説明** |
| :--- | :--- |
| --version | バージョンを表示して終了します。 |
| --help | このメッセージを表示して終了します。 |

**コマンド**
| **コマンド** | **説明** |
| :--- | :--- |
| agent | W&Bエージェントを実行 |
| artifact | アーティファクトとのやり取りのためのコマンド |
| controller | W&Bローカルスイープコントローラーを実行 |
| disabled | W&Bを無効にする。 |
| docker | コードをdockerコンテナで実行。 |
| docker-run | `docker run`をラップし、WANDB_API_KEYおよびWANDB_DOCKERを追加... |
| enabled | W&Bを有効にする。 |
| import | 他のシステムからのデータのインポートコマンド |
| init | Weights & Biasesとともにディレクトリを設定 |
| launch | W&Bジョブを開始またはキューに追加。 |
| launch-agent | W&B launchエージェントを実行。 |
| login | Weights & Biasesにログイン |
| offline | W&B syncを無効にする |
| online | W&B syncを有効にする |
| pull | Weights & Biasesからファイルをダウンロード |
| restore | runのコード、設定、およびdocker状態を復元 |
| scheduler | W&B launch スイープスケジューラーを実行（実験的） |
| server | ローカルW&Bサーバーの操作コマンド |
| status | 設定の表示 |
| sweep | スイープを作成 |
| sync | オフライントレーニングディレクトリをW&Bにアップロード |
| verify | ローカルインスタンスの確認 |