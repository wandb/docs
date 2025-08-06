---
title: wandb docker
---

**使用方法**

`wandb docker [OPTIONS] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**概要**

コードを dockerコンテナ で実行します。

W&B docker を使うと、コードを wandb が設定された docker イメージ内で実行できます。WANDB_DOCKER と WANDB_API_KEY の環境変数がコンテナに追加され、現在のディレクトリーがデフォルトで /app にマウントされます。追加の arg も渡せて、それらは `docker run` のイメージ名の前に追加されます。イメージを指定しなかった場合は、デフォルトのイメージが選ばれます。

```sh
wandb docker -v /mnt/dataset:/app/data
wandb docker gcr.io/kubeflow-images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter
wandb docker wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5"
```

デフォルトでは、エントリーポイントを上書きして wandb の有無を確認し、未インストールの場合は自動でインストールします。--jupyter フラグを渡すと、jupyter もインストールし、ポート 8888 で jupyter lab を起動します。nvidia-docker をシステム上で検知した場合は、nvidia runtime を利用します。既存の docker run コマンドで環境変数だけ wandb に設定してほしい場合は、wandb docker-run コマンドをご利用ください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--nvidia / --no-nvidia` | nvidia runtime を使用します。nvidia-docker が存在する場合はデフォルトで有効 |
| `--digest` | イメージのダイジェストを出力して終了 |
| `--jupyter / --no-jupyter` | コンテナ内で jupyter lab を起動 |
| `--dir` | コンテナ内でコードをマウントするディレクトリーを指定 |
| `--no-dir` | 現在のディレクトリーをマウントしない |
| `--shell` | コンテナ起動時に使うシェルを指定 |
| `--port` | jupyter をバインドするホストのポート番号 |
| `--cmd` | コンテナ内で実行するコマンド |
| `--no-tty` | tty なしでコマンドを実行 |