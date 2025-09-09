---
title: wandb docker
menu:
  reference:
    identifier: ja-ref-cli-wandb-docker
---

**使用方法**

`wandb docker [OPTIONS] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**概要**

dockerコンテナで コード を実行します。

W&B docker は、wandb が設定済みであることを確実にしつつ、docker イメージ内で コード を実行できるようにします。コンテナに WANDB_DOCKER と WANDB_API_KEY の 環境変数 を追加し、デフォルトで現在の ディレクトリー を /app にマウントします。イメージ名が宣言される前に `docker run` に追加される追加の arg を渡すことができます。指定がない場合は、デフォルトのイメージをこちらで選びます:

```sh
wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-
images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter wandb docker
wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5"
```

デフォルトでは、エントリポイントを上書きして wandb の存在を確認し、未インストールであればインストールします。--jupyter フラグを渡すと、Jupyter がインストールされていることを確認し、ポート 8888 で Jupyter Lab を起動します。システムに nvidia-docker があると検出した場合は、nvidia ランタイムを使用します。既存の `docker run` コマンドに対して wandb による 環境変数 の設定だけを行いたい場合は、wandb docker-run コマンドを参照してください。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--nvidia / --no-nvidia` | nvidia ランタイムを使用します。nvidia-docker が存在する場合はデフォルトで nvidia を使用します |
| `--digest` | イメージのダイジェストを出力して終了します |
| `--jupyter / --no-jupyter` | dockerコンテナ内で Jupyter Lab を実行します |
| `--dir` | コンテナ内で コード をマウントする ディレクトリー を指定します |
| `--no-dir` | 現在の ディレクトリー をマウントしません |
| `--shell` | コンテナを起動する際に使うシェル |
| `--port` | Jupyter をバインドするホストのポート |
| `--cmd` | コンテナ内で実行する コマンド |
| `--no-tty` | tty なしで コマンド を実行します |