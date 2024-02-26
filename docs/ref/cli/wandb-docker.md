# wandb docker

**使い方**

`wandb docker [オプション] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**概要**

Dockerコンテナでコードを実行します。

W&B dockerを使うと、wandbが設定された状態でDockerイメージでコードを実行できます。デフォルトでは、WANDB_DOCKERとWANDB_API_KEY環境変数をコンテナに追加し、現在のディレクトリを/appにマウントします。追加の引数を渡すと、イメージ名が宣言される前に`docker run`に追加されます。イメージが渡されない場合はデフォルトのイメージを選択します。

```sh wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter wandb docker wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5" ```

デフォルトでは、wandbの存在をチェックし、存在しない場合はインストールするためにentrypointを上書きします。--jupyterフラグを渡すと、jupyterがインストールされていることを確認し、ポート8888でjupyter labを起動します。nvidia-dockerがシステムに検出されると、nvidiaランタイムを使用します。既存のdocker runコマンドにwandbの環境変数を設定したいだけの場合は、wandb docker-runコマンドを参照してください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| --nvidia / --no-nvidia | nvidiaランタイムを使用する。nvidia-dockerが存在する場合はデフォルトでnvidiaに設定される |
| --digest | イメージのダイジェストを出力して終了 |
| --jupyter / --no-jupyter | コンテナでjupyter labを実行する |
| --dir | コンテナ内でコードをマウントするディレクトリ |
| --no-dir | 現在のディレクトリをマウントしない |
| --shell | コンテナを起動するシェル |
| --port | jupyterをバインドするホストポート |
| --cmd | コンテナで実行するコマンド |
| --no-tty | ttyなしでコマンドを実行する |
| --help |このメッセージを表示して終了する。 |