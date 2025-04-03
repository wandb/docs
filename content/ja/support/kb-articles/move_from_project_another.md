---
title: Is it possible to move a run from one project to another?
menu:
  support:
    identifier: ja-support-kb-articles-move_from_project_another
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

run を別のプロジェクトに移動するには、以下の手順に従ってください。

- 移動する run があるプロジェクトページに移動します。
- **Runs** タブをクリックして、run テーブルを開きます。
- 移動する run を選択します。
- **Move** ボタンをクリックします。
- 移動先のプロジェクトを選択し、操作を確定します。

W&B は UI を介した run の移動をサポートしていますが、run のコピーはサポートしていません。run で ログ された Artifacts は、新しいプロジェクトに転送されません。Artifacts を run の新しい場所に手動で移動するには、[`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ja" >}}) SDK コマンドまたは [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ja" >}}) を使用して Artifacts をダウンロードし、[wandb artifact put]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ja" >}}) または `Api.artifact` API を使用して、run の新しい場所にアップロードします。
