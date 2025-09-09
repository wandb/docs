---
title: runs を移動
menu:
  default:
    identifier: ja-guides-models-track-runs-manage-runs
    parent: what-are-runs
---

このページでは、run をある project から別の project へ、team の内外へ、またはある team から別の team へ移動する方法を説明します。現在の場所と移動先の両方でその run への アクセス 権が必要です。

{{% alert %}}
run を移動しても、関連する過去の Artifacts は移動されません。Artifact を手動で移動するには、[`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ja" >}}) SDK のコマンドまたは [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ja" >}}) を使用して Artifact をダウンロードし、[`wandb artifact put`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ja" >}}) もしくは `Api.artifact` API を使用して run の新しい場所にアップロードします。
{{% /alert %}}

Runs タブをカスタマイズするには、[Project page]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}}) を参照してください。

runs を experiments にグループ化している場合は、[Set a group in the UI]({{< relref path="grouping.md#set-a-group-in-the-ui" lang="ja" >}}) を参照してください。

## 自分の projects 間で runs を移動する

runs をある project から別の project に移動するには:

1. 移動したい runs を含む project に移動します。
2. project のサイドバーから Runs タブを選択します。
3. 移動したい runs の横にあるチェックボックスを選択します。
4. テーブル上部の Move ボタンを選択します。
5. ドロップダウンから移動先の project を選択します.

{{< img src="/images/app_ui/howto_move_runs.gif" alt="projects 間で run を移動するデモ" >}}

## runs を team に移動する

あなたがメンバーである team に runs を移動するには:

1. 移動したい runs を含む project に移動します。
2. project のサイドバーから Runs タブを選択します。
3. 移動したい runs の横にあるチェックボックスを選択します。
4. テーブル上部の Move ボタンを選択します。
5. ドロップダウンから移動先の team と project を選択します。

{{< img src="/images/app_ui/demo_move_runs.gif" alt="run を team に移動するデモ" >}}