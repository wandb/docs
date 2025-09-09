---
title: Launch ジョブを表示
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-launch-view-jobs
    parent: create-and-deploy-jobs
url: guides/launch/launch-view-jobs
---

このページでは、キューに追加された Launch ジョブに関する情報の見方を説明します。

## ジョブを表示

W&B App で、キューに追加されたジョブを表示します。

1. https://wandb.ai/home の W&B App にアクセスします。
2. 左サイドバーの Applications セクションで Launch を選択します。
3. All entities ドロップダウンを開き、その Launch ジョブが属する Entity を選択します。
4. Launch Application ページで折りたたみ UI を展開し、その特定のキューに追加されたジョブの一覧を表示します。

{{% alert %}}
Launch エージェントが Launch ジョブを実行すると run が作成されます。つまり、一覧に表示される各 run は、そのキューに追加された特定のジョブに対応しています。
{{% /alert %}}

例えば、次の画像は `job-source-launch_demo-canonical` というジョブから作成された 2 つの run を示しています。ジョブは `Start queue` というキューに追加されました。最初の run は `resilient-snowball`、2 つ目の run は `earthy-energy-165` です。

{{< img src="/images/launch/launch_jobs_status.png" alt="Launch jobs status view" >}}

W&B App の UI では、Launch ジョブから作成された run について、次の追加情報を確認できます:
   - **Run**: そのジョブに割り当てられた W&B の run 名。
   - **Job ID**: ジョブの名前。
   - **Project**: run が属する Project の名前。
   - **Status**: キューに入っている run のステータス。
   - **Author**: その run を作成した W&B の Entity。
   - **Creation date**: キューが作成されたタイムスタンプ。
   - **Start time**: ジョブの開始時刻のタイムスタンプ。
   - **Duration**: ジョブの run が完了するまでに要した時間（秒）。

## ジョブを一覧表示
W&B CLI で、特定の Project に存在するジョブの一覧を表示できます。W&B の job list コマンドを使い、その Launch ジョブが属する Project 名と Entity 名を、それぞれ `--project` と `--entity` フラグに指定します。

```bash
 wandb job list --entity your-entity --project project-name
```

## ジョブのステータスを確認

次の表は、キューに入っている run にあり得るステータスの定義です:

| ステータス | 説明 |
| --- | --- |
| **Idle** | run が、アクティブなエージェントが 1 つもいないキューにあります。 |
| **Queued** | run が、エージェントによる処理を待ってキューにあります。 |
| **Pending** | run はエージェントに取得されましたが、まだ開始していません。クラスター上のリソース不足などが原因の可能性があります。 |
| **Running** | run は現在実行中です。 |
| **Killed** | ジョブがユーザーによって強制終了されました。 |
| **Crashed** | run がデータの送信を停止した、または正常に開始できませんでした。 |
| **Failed** | run が非ゼロの終了コードで終了した、または開始に失敗しました。 |
| **Finished** | ジョブが正常に完了しました。 |