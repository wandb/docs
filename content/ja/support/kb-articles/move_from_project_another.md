---
title: 1 つの Project から別の Project に run を移動することは可能ですか？
menu:
  support:
    identifier: ja-support-kb-articles-move_from_project_another
support:
- runs
toc_hide: true
type: docs
url: /ja/support/:filename
---

run をあるプロジェクトから別のプロジェクトに移動するには、次の手順に従います：

- 移動する run があるプロジェクトページに移動します。
- **Runs** タブをクリックして、run のテーブルを開きます。
- 移動する run を選択します。
- **Move** ボタンをクリックします。
- 移動先のプロジェクトを選択し、操作を確認します。

W&B は UI を介して run の移動をサポートしていますが、run のコピーはサポートしていません。run にログされた アーティファクト は新しいプロジェクトには転送されません。run の新しい場所にアーティファクト を手動で移動するには、[`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ja" >}}) SDK コマンドまたは [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ja" >}}) を使用してアーティファクトをダウンロードし、[wandb artifact put]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ja" >}}) または `Api.artifact` API を使用して run の新しい場所にアップロードできます。