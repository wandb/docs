---
title: Where are artifacts downloaded, and how can I control that?
menu:
  support:
    identifier: ko-support-kb-articles-where_are_artifacts_downloaded_and_how_can_i_control_that
support:
- artifacts
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

기본적으로 아티팩트는 `artifacts/` 폴더에 다운로드됩니다. 위치를 변경하려면 다음을 수행하세요.

- [`wandb.Artifact().download`]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}) 에 전달합니다.

    ```python
    wandb.Artifact().download(root="<path_to_download>")
    ```

- `WANDB_ARTIFACT_DIR` [환경 변수]({{< relref path="guides/models/track/environment-variables.md" lang="ko" >}}) 를 설정합니다.

    ```python
    import os
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```
