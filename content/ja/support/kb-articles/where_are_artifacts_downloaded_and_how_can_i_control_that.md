---
menu:
  support:
    identifier: ja-support-kb-articles-where_are_artifacts_downloaded_and_how_can_i_control_that
support:
- artifacts
- environment variables
title: Where are artifacts downloaded, and how can I control that?
toc_hide: true
type: docs
url: /support/:filename
---

By default, artifacts download to the `artifacts/` folder. To change the location:

- Pass it to [`wandb.Artifact().download`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}):

    ```python
    wandb.Artifact().download(root="<path_to_download>")
    ```

- Set the `WANDB_ARTIFACT_DIR` [environment variable]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}):

    ```python
    import os
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```