---
title: アーティファクトはどこにダウンロードされますか？また、その保存場所をどのように指定できますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
- 環境変数
---

デフォルトでは、アーティファクトは `artifacts/` フォルダにダウンロードされます。ダウンロード先を変更するには、以下の方法があります。

- [`wandb.Artifact().download`]({{< relref "/ref/python/public-api/api.md" >}}) にパスを指定します。

    ```python
    wandb.Artifact().download(root="<path_to_download>")
    ```

- `WANDB_ARTIFACT_DIR` [環境変数]({{< relref "guides/models/track/environment-variables.md" >}}) を設定します。

    ```python
    import os
    # ダウンロード先のパスを指定
    os.environ["WANDB_ARTIFACT_DIR"] = "<path_to_download>"
    ```