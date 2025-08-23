---
title: 아티팩트는 어디에 다운로드되며, 이 위치를 어떻게 제어할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-where_are_artifacts_downloaded_and_how_can_i_control_that
support:
- 아티팩트
- 환경 변수
toc_hide: true
type: docs
url: /support/:filename
---

기본적으로, Artifacts 는 `artifacts/` 폴더에 다운로드됩니다. 위치를 변경하려면 아래 방법을 사용하세요:

- [`wandb.Artifact().download`]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}}) 에서 위치를 지정합니다:

    ```python
    # 다운로드 경로를 지정하여 Artifacts를 다운로드합니다.
    wandb.Artifact().download(root="<path_to_download>")
    ```

- `WANDB_ARTIFACT_DIR` [환경 변수]({{< relref path="guides/models/track/environment-variables.md" lang="ko" >}})를 설정합니다:

    ```python
    import os
    # Artifacts가 다운로드될 경로를 환경 변수로 지정합니다.
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```