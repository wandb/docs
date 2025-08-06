---
title: wandb docker-run
menu:
  reference:
    identifier: ja-ref-cli-wandb-docker-run
---

**使用方法**

`wandb docker-run [OPTIONS] [DOCKER_RUN_ARGS]...`

**概要**

`docker run` をラップし、WANDB_API_KEY および WANDB_DOCKER 環境変数を追加します。

nvidia-docker 実行ファイルがシステム上に存在し、--runtime が指定されていない場合は、ランタイムも nvidia に設定されます。

詳細は `docker run --help` をご覧ください。


**オプション**

| **オプション** | **説明** |
| :--- | :--- |