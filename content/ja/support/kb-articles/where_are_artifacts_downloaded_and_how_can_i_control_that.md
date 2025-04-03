---
title: Where are artifacts downloaded, and how can I control that?
menu:
  support:
    identifier: ja-support-kb-articles-where_are_artifacts_downloaded_and_how_can_i_control_that
support:
- artifacts
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

デフォルトでは、Artifacts は `artifacts/` フォルダーにダウンロードされます。場所を変更するには:

- [`wandb.Artifact().download`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) に渡します。

    ```python
    wandb.Artifact().download(root="<path_to_download>")
    ```

- `WANDB_ARTIFACT_DIR` [環境変数]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}) を設定します。

    ```python
    import os
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```
