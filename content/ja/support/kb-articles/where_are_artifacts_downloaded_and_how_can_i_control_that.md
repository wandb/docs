---
title: Artifacts はどこにダウンロードされ、どのように管理できますか？
menu:
  support:
    identifier: ja-support-kb-articles-where_are_artifacts_downloaded_and_how_can_i_control_that
support:
- アーティファクト
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

デフォルトでは、 artifacts は `artifacts/` フォルダにダウンロードされます。保存先を変更するには:

- [`wandb.Artifact().download`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) に渡す:

    ```python
    wandb.Artifact().download(root="<path_to_download>")
    ```

- `WANDB_ARTIFACT_DIR` [環境変数]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}) を設定する:

    ```python
    import os
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```