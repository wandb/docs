
# wandb docker

**使用方法**

`wandb docker [OPTIONS] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**概要**

コードをdockerコンテナで実行します。

W&B dockerを使用すると、コードをdockerイメージ内で実行し、wandbが設定されていることを確認できます。環境変数WANDB_DOCKERとWANDB_API_KEYをコンテナに追加し、デフォルトで現在のディレクトリーを /app にマウントします。また、名前を宣言する前に`docker run`に追加される追加のargsを渡すことができます。イメージが渡されない場合は、デフォルトのイメージを選択します。

```sh wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-
images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter wandb docker
wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5" ```

デフォルトでは、wandbの存在を確認し、インストールされていない場合はインストールするためにエントリーポイントを上書きします。--jupyterフラグを渡すと、jupyterがインストールされていることを確認し、ポート8888でjupyter labを開始します。システムでnvidia-dockerを検出した場合は、nvidiaのランタイムを使用します。既存のdocker実行コマンドに環境変数を設定するだけで良い場合は、wandb docker-run コマンドを参照してください。

**オプション**

| **Option** | **Description** |
| :--- | :--- |
| --nvidia / --no-nvidia | nvidiaランタイムを使用します。nvidia-dockerが存在する場合、デフォルトでnvidiaになります |
| --digest | イメージのダイジェストを出力して終了します |
| --jupyter / --no-jupyter | コンテナ内でjupyter labを実行します |
| --dir | コンテナ内にマウントするディレクトリー |
| --no-dir | 現在のディレクトリーをマウントしない |
| --shell | コンテナで開始するシェル |
| --port | jupyterをバインドするホストポート |
| --cmd | コンテナ内で実行するコマンド |
| --no-tty | ttyなしでコマンドを実行します |
