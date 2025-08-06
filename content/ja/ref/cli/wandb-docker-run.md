---
title: wandb docker-run
---

**使い方**

`wandb docker-run [OPTIONS] [DOCKER_RUN_ARGS]...`

**概要**

`docker run` をラップし、WANDB_API_KEY と WANDB_DOCKER 環境変数を追加します。

また、システムに nvidia-docker 実行ファイルが存在し、--runtime が指定されていない場合は、ランタイムが nvidia に設定されます。

詳細は `docker run --help` をご覧ください。


**オプション**

| **オプション** | **説明** |
| :--- | :--- |