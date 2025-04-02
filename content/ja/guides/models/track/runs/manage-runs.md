---
title: Move runs
menu:
  default:
    identifier: ja-guides-models-track-runs-manage-runs
    parent: what-are-runs
---

このページでは、ある run を別の project へ、team の内外へ、またはある team から別の team へ移動する方法について説明します。現在の場所と新しい場所で run にアクセスできる必要があります。

{{% alert %}}
run を移動しても、それに関連付けられた過去の Artifacts は移動されません。artifact を手動で移動するには、[`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ja" >}}) SDK コマンドまたは [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ja" >}}) を使用して artifact をダウンロードし、[wandb artifact put]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ja" >}}) または `Api.artifact` API を使用して run の新しい場所にアップロードします。
{{% /alert %}}

**Runs** タブをカスタマイズするには、[Project page]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}})を参照してください。

## project 間で run を移動する

ある project から別の project へ run を移動するには:

1. 移動する run が含まれている project に移動します。
2. project のサイドバーから **Runs** タブを選択します。
3. 移動する run の横にあるチェックボックスをオンにします。
4. テーブルの上にある **Move** ボタンを選択します。
5. ドロップダウンから移動先の project を選択します。

{{< img src="/images/app_ui/howto_move_runs.gif" alt="" >}}

## team へ run を移動する

自分がメンバーである team へ run を移動するには:

1. 移動する run が含まれている project に移動します。
2. project のサイドバーから **Runs** タブを選択します。
3. 移動する run の横にあるチェックボックスをオンにします。
4. テーブルの上にある **Move** ボタンを選択します。
5. ドロップダウンから移動先の team と project を選択します。

{{< img src="/images/app_ui/demo_move_runs.gif" alt="" >}}
