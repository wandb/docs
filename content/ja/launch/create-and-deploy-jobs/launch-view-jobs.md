---
title: ローンチジョブを表示
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-launch-view-jobs
    parent: create-and-deploy-jobs
url: guides/launch/launch-view-jobs
---

以下のページでは、キューに追加されたローンチジョブの情報の閲覧方法について説明します。

## ジョブの表示

W&B App を使って、キューに追加されたジョブを表示できます。

1. https://wandb.ai/home にアクセスします。
2. 左サイドバーの **Applications** セクションから **Launch** を選択します。
3. **All entities** ドロップダウンから、ローンチジョブが属する Entity を選択します。
4. Launch Application ページから折りたたみ UI を展開すると、そのキューに追加されたジョブ一覧が表示されます。

{{% alert %}}
ローンチ エージェントがローンチジョブを実行すると、run が作成されます。つまり、表示されている各 run は、そのキューに追加された特定のジョブに対応しています。
{{% /alert %}}

例えば、次の画像は `job-source-launch_demo-canonical` というジョブから作成された 2 つの run を示しています。このジョブは `Start queue` というキューに追加されました。最初にリストされている run の名前は `resilient-snowball`、2 番目は `earthy-energy-165` です。

{{< img src="/images/launch/launch_jobs_status.png" alt="Launch jobs status view" >}}

W&B App の UI 上では、ローンチジョブから作成された run について、以下のような追加情報も確認できます。

   - **Run**: そのジョブに割り当てられた W&B run の名前。
   - **Job ID**: ジョブ名。
   - **Project**: run が属する project 名。
   - **Status**: キュー内の run のステータス。
   - **Author**: run を作成した W&B entity。
   - **Creation date**: キューが作成された日時。
   - **Start time**: ジョブが開始された日時。
   - **Duration**: ジョブの run が完了するまでにかかった秒数。

## ジョブの一覧取得
W&B CLI を使って、プロジェクト内に存在するジョブの一覧を確認できます。`wandb job list` コマンドを使用し、`--project` と `--entity` フラグに、それぞれローンチジョブが属する Project 名と Entity 名を入力してください。

```bash
 wandb job list --entity your-entity --project project-name
```

## ジョブのステータス確認

次の表は、キューに入った run のステータスについて説明しています。

| ステータス | 説明 |
| --- | --- |
| **Idle** | run がキューにあり、アクティブなエージェントがいません。 |
| **Queued** | run がキューにあり、エージェントが処理するのを待っています。 |
| **Pending** | run がエージェントに取得されたものの、まだ開始していません。これはクラスターでリソースが不足している場合などに発生します。 |
| **Running** | run が現在実行中です。 |
| **Killed** | ジョブがユーザーによって強制終了されました。 |
| **Crashed** | run がデータの送信を停止した、または正常に開始できませんでした。 |
| **Failed** | run がゼロ以外の終了コードで終了、または開始に失敗しました。 |
| **Finished** | ジョブが正常に完了しました。 |