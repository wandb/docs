---
title: ローンチ ジョブを表示
menu:
  launch:
    identifier: launch-view-jobs
    parent: create-and-deploy-jobs
url: guides/launch/launch-view-jobs
---

以下のページでは、キューに追加されたローンチジョブに関する情報の表示方法について説明します。

## ジョブの表示

W&B App を使って、キューに追加されたジョブを表示できます。

1. https://wandb.ai/home にアクセスします。
2. 左側のサイドバーの **Applications** セクションから **Launch** を選択します。
3. **All entities** のドロップダウンを選択し、そのローンチジョブが属している entity を選びます。
4. Launch Application ページで UI を展開し、そのキューに追加されたジョブの一覧を表示します。

{{% alert %}}
ローンチエージェントがローンチジョブを実行すると、run が作成されます。つまり、そのキューに追加された特定のジョブごとに 1 つの run が表示されます。
{{% /alert %}}

例えば、下の画像では `job-source-launch_demo-canonical` というジョブから作成された 2 つの run が確認できます。このジョブは `Start queue` というキューに追加されました。最初の run はキュー `resilient-snowball` にあり、2 番目の run は `earthy-energy-165` です。

{{< img src="/images/launch/launch_jobs_status.png" alt="Launch jobs status view" >}}

W&B App の UI では、ローンチジョブから作成された run についてさらなる情報が確認できます。例えば下記の通りです。
   - **Run**: そのジョブに割り当てられた W&B run の名前
   - **Job ID**: ジョブの名前
   - **Project**: run が属するプロジェクト名
   - **Status**: キューにある run のステータス
   - **Author**: run を作成した W&B entity
   - **Creation date**: キューが作成された日時
   - **Start time**: ジョブの開始日時
   - **Duration**: ジョブの run 完了までにかかった秒数

## ジョブ一覧の取得 
W&B CLI を用いて、プロジェクト内に存在するジョブの一覧を表示できます。W&B のジョブ一覧コマンドを使用し、`--project` と `--entity` フラグにそれぞれプロジェクト名と entity 名を指定してください。

```bash
 wandb job list --entity your-entity --project project-name
```

## ジョブのステータス確認

次の表は、キューにある run が持つ可能性のあるステータスを示します。

| ステータス | 説明 |
| --- | --- |
| **Idle** | run はアクティブなエージェントのいないキューにあります。 |
| **Queued** | run はエージェントによる処理待ちの状態でキューにあります。|
| **Pending** | run はエージェントによってピックアップされましたが、まだ開始されていません。この状態はクラスター上でリソースが足りない場合などに発生します。|
| **Running** | run は現在実行中です。|
| **Killed** | ジョブがユーザーによって強制終了されました。|
| **Crashed** | run がデータ送信を停止した、または正常に開始しませんでした。|
| **Failed** | run が非ゼロの終了コードで終了した、または開始に失敗しました。|
| **Finished** | ジョブが正常に完了しました。|