---
title: アーティファクトはどこにダウンロードされ、どのように保存先を指定できますか？
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

デフォルトでは、Artifacts は `artifacts/` フォルダにダウンロードされます。場所を変更するには、以下の方法があります。

- [`wandb.Artifact().download`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) にパスを指定する:

    ```python
    wandb.Artifact().download(root="<path_to_download>")
    ```

- `WANDB_ARTIFACT_DIR` [環境変数]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}) を設定する:

    ```python
    import os
    # Artifact のダウンロード先を指定
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```