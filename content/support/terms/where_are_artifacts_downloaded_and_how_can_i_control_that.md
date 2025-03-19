---
title: Where are artifacts downloaded, and how can I control that?
toc_hide: true
type: docs
support:
  - artifacts
  - environment variables
---

By default, artifacts download to the `artifacts/` folder. To change the location:

- Pass it to [`wandb.Artifact().download`]({{< relref "/ref/python/public-api/api.md" >}}):

    ```python
    wandb.Artifact().download(root="<path_to_download>")
    ```

- Set the `WANDB_ARTIFACT_DIR` [environment variable]({{< relref "guides/models/track/environment-variables.md" >}}):

    ```python
    import os
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```