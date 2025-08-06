---
title: run を別のプロジェクトに移動することは可能ですか？
menu:
  support:
    identifier: ja-support-kb-articles-move_from_project_another
support:
- run
toc_hide: true
type: docs
url: /support/:filename
---

run を別の Project に移動するには、以下の手順に従ってください。

- 移動したい run がある Project ページに移動します。
- **Runs** タブをクリックして runs テーブルを開きます。
- 移動したい run を選択します。
- **Move** ボタンをクリックします。
- 移動先の Project を選択し、操作を確定します。

W&B は UI を通じて run の移動をサポートしていますが、run のコピーはサポートしていません。run と一緒にログされた Artifacts は新しい Project には移動されません。Artifacts を手動で run の新しい場所に移動するには、[`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ja" >}}) SDK コマンドや、[`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ja" >}}) を使って Artifact をダウンロードし、その後 [wandb artifact put]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ja" >}}) や `Api.artifact` API を使って新しい run の場所にアップロードしてください。