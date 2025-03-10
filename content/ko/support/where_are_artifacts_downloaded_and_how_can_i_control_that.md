---
menu:
  support:
    identifier: ko-support-where_are_artifacts_downloaded_and_how_can_i_control_that
tags:
- artifacts
- environment variables
title: Where are artifacts downloaded, and how can I control that?
toc_hide: true
type: docs
---

By default, artifacts download to the `artifacts/` folder. To change the location:

- Pass it to [`wandb.Artifact().download`]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}):

    ```python
    wandb.Artifact().download(root="<path_to_download>")
    ```

- Set the `WANDB_ARTIFACT_DIR` [environment variable]({{< relref path="guides/models/track/environment-variables.md" lang="ko" >}}):

    ```python
    import os
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```