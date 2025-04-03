---
title: wandb docker
menu:
  reference:
    identifier: ja-ref-cli-wandb-docker
---

**使用方法**

`wandb docker [OPTIONS] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**概要**

dockerコンテナ内でコードを実行します。

W&B docker を使用すると、wandb が確実に構成されるように、docker イメージでコードを実行できます。WANDB_DOCKER と WANDB_API_KEY の 環境 変数をコンテナに追加し、デフォルトで現在の ディレクトリー を /app にマウントします。イメージ名が宣言される前に `docker run` に追加される追加の arg を渡すことができます。イメージが渡されない場合は、デフォルトのイメージを選択します。

```sh wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-
images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter wandb docker
wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5" ```

デフォルトでは、wandb の存在を確認し、存在しない場合はインストールするために、エントリポイントをオーバーライドします。--jupyter フラグを渡すと、jupyter がインストールされていることを確認し、ポート 8888 で jupyter lab を起動します。システムで nvidia-docker が検出された場合は、nvidia ランタイムを使用します。既存の docker run コマンドに 環境 変数を設定するために wandb のみが必要な場合は、wandb docker-run コマンドを参照してください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--nvidia / --no-nvidia` | nvidia ランタイムを使用します。nvidia-docker が存在する場合は、デフォルトで nvidia になります。 |
| `--digest` | イメージ ダイジェストを出力して終了します |
| `--jupyter / --no-jupyter` | コンテナ内で jupyter lab を実行します |
| `--dir` | コンテナ内のコードをマウントする ディレクトリー |
| `--no-dir` | 現在の ディレクトリー をマウントしません |
| `--shell` | コンテナの起動に使用するシェル |
| `--port` | jupyter をバインドするホスト ポート |
| `--cmd` | コンテナで実行する コマンド |
| `--no-tty` | tty なしで コマンド を実行します |
