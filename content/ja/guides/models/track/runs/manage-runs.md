---
title: run を移動する
menu:
  default:
    identifier: manage-runs
    parent: what-are-runs
---

このページでは、run をある Project から別の Project へ、Team への追加や移動、あるいは Team 間で移動する方法をご紹介します。現在の場所および新しい場所の両方で run へのアクセス権が必要です。

{{% alert %}}
run を移動しても、それに関連付けられている過去の Artifacts は自動的には移動されません。Artifacts を手動で移動する場合は、[`wandb artifact get`]({{< relref "/ref/cli/wandb-artifact/wandb-artifact-get/" >}}) SDK コマンドや [`Api.artifact` API]({{< relref "/ref/python/public-api/api/#artifact" >}}) でアーティファクトをダウンロードし、[`wandb artifact put`]({{< relref "/ref/cli/wandb-artifact/wandb-artifact-put/" >}}) または `Api.artifact` API を使って新しい場所にアップロードしてください。
{{% /alert %}}

**Runs** タブのカスタマイズについては、[Project page]({{< relref "/guides/models/track/project-page.md#runs-tab" >}}) を参照してください。

run を Experiments ごとにまとめている場合は、[Set a group in the UI]({{< relref "grouping.md#set-a-group-in-the-ui" >}}) を参照してください。

## Project 間で run を移動する

run をある Project から別の Project へ移動する手順:

1. 移動したい run が含まれている Project に移動します。
2. Project サイドバーから **Runs** タブを選択します。
3. 移動したい run の横にあるチェックボックスを選択します。
4. テーブル上部の **Move** ボタンをクリックします。
5. プルダウンから移動先の Project を選択します。

{{< img src="/images/app_ui/howto_move_runs.gif" alt="Demo of moving a run between projects" >}}

## run を Team に移動する

自分がメンバーになっている Team へ run を移動するには次の通りです。

1. 移動したい run が含まれている Project に移動します。
2. Project サイドバーから **Runs** タブを選択します。
3. 移動したい run の横にあるチェックボックスを選択します。
4. テーブル上部の **Move** ボタンをクリックします。
5. プルダウンから移動先の Team と Project を選択します。

{{< img src="/images/app_ui/demo_move_runs.gif" alt="Demo of moving a run to a team" >}}