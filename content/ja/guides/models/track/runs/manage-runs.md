---
title: run を移動する
menu:
  default:
    identifier: ja-guides-models-track-runs-manage-runs
    parent: what-are-runs
---

このページでは、run をある Project から別の Project へ、Team 内や Team 間で移動する方法を説明します。現在の場所と新しい場所の両方で run へのアクセス権が必要です。

{{% alert %}}
run を移動しても、それに関連付けられている過去の Artifacts は移動されません。Artifacts を手動で移動したい場合は、[`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ja" >}}) SDK コマンドや [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ja" >}}) を使って Artifact をダウンロードし、その後 [`wandb artifact put`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ja" >}}) または `Api.artifact` API を使って新しい場所へアップロードしてください。
{{% /alert %}}

**Runs** タブのカスタマイズについては [Project page]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}}) をご参照ください。

run を experiment のグループとしてまとめる場合は、[Set a group in the UI]({{< relref path="grouping.md#set-a-group-in-the-ui" lang="ja" >}}) をご参照ください。

## Project 間で run を移動する

run をある Project から別の Project へ移動するには：

1. 移動したい run が含まれている Project へ移動します。
2. Project サイドバーから **Runs** タブを選択します。
3. 移動したい run の横にあるチェックボックスを選択します。
4. テーブル上部の **Move** ボタンをクリックします。
5. ドロップダウンから移動先の Project を選択します。

{{< img src="/images/app_ui/howto_move_runs.gif" alt="Project 間で run を移動するデモ" >}}

## run を Team へ移動する

自分がメンバーである Team に run を移動するには：

1. 移動したい run が含まれている Project へ移動します。
2. Project サイドバーから **Runs** タブを選択します。
3. 移動したい run の横にあるチェックボックスを選択します。
4. テーブル上部の **Move** ボタンをクリックします。
5. ドロップダウンから移動先の Team と Project を選択します。

{{< img src="/images/app_ui/demo_move_runs.gif" alt="run を Team へ移動するデモ" >}}