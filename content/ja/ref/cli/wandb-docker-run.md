---
title: wandb docker-run
menu:
  reference:
    identifier: ja-ref-cli-wandb-docker-run
---

**使用方法**

`wandb docker-run [OPTIONS] [DOCKER_RUN_ARGS]...`

**概要**

`docker run` をラップし、環境変数 WANDB_API_KEY と WANDB_DOCKER を追加します。

システムに nvidia-docker の 実行可能ファイルが存在し、かつ --runtime が設定されていない場合、runtime を nvidia に設定します。

詳細は `docker run --help` を参照してください。


**オプション**

| **オプション** | **説明** |
| :--- | :--- |