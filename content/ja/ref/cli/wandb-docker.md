---
title: wandb docker
menu:
  reference:
    identifier: ja-ref-cli-wandb-docker
---

**使い方**

`wandb docker [OPTIONS] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**概要**

あなたのコードを docker コンテナで実行します。

W&B docker を使うと、wandb が設定された状態で docker イメージ内でコードを実行できます。自動的に `WANDB_DOCKER` と `WANDB_API_KEY` の環境変数をコンテナに追加し、現在のディレクトリーをデフォルトで /app にマウントします。追加で arg を渡すことができ、これらはイメージ名の前に `docker run` に追加されます。イメージ名を指定しない場合はデフォルトイメージを選びます。

```sh
# データセットをマウントして docker コンテナを起動
wandb docker -v /mnt/dataset:/app/data

# jupyter を使った docker イメージを起動
wandb docker gcr.io/kubeflow-images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter

# keras-gpu イメージでスクリプトを実行
wandb docker wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5"
```

デフォルトでは wandb の存在を確認し、未インストールの場合は自動でインストールするため entrypoint を上書きします。`--jupyter` フラグを指定すると、jupyter をインストールし、ポート 8888 で jupyter lab を開始します。システム上で nvidia-docker を検出した場合、自動的に nvidia runtime を利用します。既存の docker run コマンドに対して wandb が環境変数だけを設定したい場合は、`wandb docker-run` コマンドをご利用ください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--nvidia / --no-nvidia` | nvidia runtime を使用します。nvidia-docker が存在する場合はデフォルトで有効です。 |
| `--digest` | イメージのダイジェストを出力して終了します。 |
| `--jupyter / --no-jupyter` | コンテナ内で jupyter lab を実行します。 |
| `--dir` | コンテナ内でコードをマウントするディレクトリーを指定します。 |
| `--no-dir` | 現在のディレクトリーをマウントしません。 |
| `--shell` | コンテナ起動時に使用するシェルを指定します。 |
| `--port` | jupyter をバインドするホストのポート番号を指定します。 |
| `--cmd` | コンテナ内で実行するコマンドを指定します。 |
| `--no-tty` | tty なしでコマンドを実行します。 |