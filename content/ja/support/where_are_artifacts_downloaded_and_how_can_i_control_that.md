---
title: Where are artifacts downloaded, and how can I control that?
menu:
  support:
    identifier: ja-support-where_are_artifacts_downloaded_and_how_can_i_control_that
tags:
- artifacts
- environment variables
toc_hide: true
type: docs
---

デフォルトでは、アーティファクトは `artifacts/` フォルダにダウンロードされます。場所を変更するには以下の方法があります：

- [`wandb.Artifact().download`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) に渡す：

    ```python
    wandb.Artifact().download(root="<path_to_download>")
    ```

- `WANDB_ARTIFACT_DIR` [環境変数]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}) を設定する：

    ```python
    import os
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```