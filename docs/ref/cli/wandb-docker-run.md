# wandb docker-run

**使用方法**

`wandb docker-run [オプション] [DOCKER_RUN_ARG]...`

**概要**

`docker run`をラップし、WANDB_API_KEYとWANDB_DOCKER環境変数を追加します。

また、nvidia-docker実行ファイルがシステムに存在し、--runtimeが設定されていない場合、ランタイムをnvidiaに設定します。

詳細は`docker run --help`を参照してください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| --help | このメッセージを表示して終了します。 |