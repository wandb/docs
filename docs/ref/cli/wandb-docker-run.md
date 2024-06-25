
# wandb docker-run

**使用方法**

`wandb docker-run [OPTIONS] [DOCKER_RUN_ARGS]...`

**概要**

`docker run` をラップし、WANDB_API_KEY および WANDB_DOCKER 環境変数を追加します。

このコマンドは、システムに nvidia-docker 実行可能ファイルが存在し、--runtime が設定されていない場合、ランタイムを nvidia に設定します。

詳細は `docker run --help` を参照してください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |