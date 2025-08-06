---
title: run をある Project から別の Project に移動することは可能ですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- run
---

run を別の Project に移動するには、以下の手順を実行してください。

- 移動したい run がある Project ページに移動します。
- **Runs** タブをクリックして run の一覧テーブルを開きます。
- 移動する run を選択します。
- **Move** ボタンをクリックします。
- 移動先の Project を選択し、操作を確定します。

W&B では、UI を使って run の移動はサポートされていますが、run のコピーはサポートされていません。run とともに記録された Artifacts は新しい Project には自動的に移動されません。Artifacts を手動で新しい run の場所に移動するには、[`wandb artifact get`]({{< relref "/ref/cli/wandb-artifact/wandb-artifact-get/" >}}) SDK コマンドや、[`Api.artifact` API]({{< relref "/ref/python/public-api/api/#artifact" >}}) でアーティファクトをダウンロードし、その後 [wandb artifact put]({{< relref "/ref/cli/wandb-artifact/wandb-artifact-put/" >}}) や `Api.artifact` API を使って新しい場所にアップロードしてください。