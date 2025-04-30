---
title: run を移動する
menu:
  default:
    identifier: ja-guides-models-track-runs-manage-runs
    parent: what-are-runs
---

このページでは、run を別のプロジェクト間で、またはチーム内外、またはあるチームから別のチームへの移動方法を示します。現在の場所と新しい場所の両方で run へのアクセス権が必要です。

{{% alert %}}
run を移動する際、関連する履歴アーティファクトは移動されません。アーティファクトを手動で移動するには、[`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ja" >}}) SDK コマンドや [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ja" >}}) を使用してアーティファクトをダウンロードしてから、[wandb artifact put]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ja" >}}) や `Api.artifact` API を使用して、run の新しい場所にアップロードします。
{{% /alert %}}

**Runs** タブをカスタマイズするには、[Project page]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ja" >}}) を参照してください。

## プロジェクト間で run を移動する

run をあるプロジェクトから別のプロジェクトに移動するには:

1. 移動したい run を含むプロジェクトに移動します。
2. プロジェクトのサイドバーから **Runs** タブを選択します。
3. 移動したい run の横にあるチェックボックスを選択します。
4. テーブルの上にある **Move** ボタンを選択します。
5. ドロップダウンから移動先のプロジェクトを選択します。

{{< img src="/images/app_ui/howto_move_runs.gif" alt="" >}}

## チームに run を移動する

あなたがメンバーであるチームに run を移動するには:

1. 移動したい run を含むプロジェクトに移動します。
2. プロジェクトのサイドバーから **Runs** タブを選択します。
3. 移動したい run の横にあるチェックボックスを選択します。
4. テーブルの上にある **Move** ボタンを選択します。
5. ドロップダウンから移動先のチームとプロジェクトを選択します。

{{< img src="/images/app_ui/demo_move_runs.gif" alt="" >}}