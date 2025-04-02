---
title: View launch jobs
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-launch-view-jobs
    parent: create-and-deploy-jobs
url: guides/launch/launch-view-jobs
---

以下のページでは、キューに追加された Launch ジョブに関する情報を表示する方法について説明します。

## ジョブの表示

W&B アプリでキューに追加されたジョブを表示します。

1. W&B アプリ (https://wandb.ai/home) に移動します。
2. 左側のサイドバーの [**Applications**] セクションで [**Launch**] を選択します。
3. [**All entities**] ドロップダウンを選択し、Launch ジョブが属するエンティティを選択します。
4. Launch アプリケーションページから折りたたみ可能なUIを展開して、その特定のキューに追加されたジョブのリストを表示します。

{{% alert %}}
Launch エージェントが Launch ジョブを実行すると、run が作成されます。言い換えれば、リストされている各 run は、そのキューに追加された特定のジョブに対応します。
{{% /alert %}}

たとえば、次の図は、`job-source-launch_demo-canonical` というジョブから作成された2つの run を示しています。このジョブは `Start queue` というキューに追加されました。キューにリストされている最初の run は `resilient-snowball` と呼ばれ、2番目の run は `earthy-energy-165` と呼ばれます。

{{< img src="/images/launch/launch_jobs_status.png" alt="" >}}

W&B アプリのUI内では、Launch ジョブから作成された run に関する追加情報を見つけることができます。
   - **Run**: そのジョブに割り当てられた W&B の run の名前。
   - **Job ID**: ジョブの名前。
   - **Project**: run が属する project の名前。
   - **Status**: キューに入れられた run のステータス。
   - **Author**: run を作成した W&B エンティティ。
   - **Creation date**: キューが作成されたときのタイムスタンプ。
   - **Start time**: ジョブが開始されたときのタイムスタンプ。
   - **Duration**: ジョブの run が完了するまでにかかった時間（秒単位）。

## ジョブのリスト表示
W&B CLI を使用して、project 内に存在するジョブのリストを表示します。W&B job list コマンドを使用し、Launch ジョブが属する project とエンティティの名前をそれぞれ `--project` および `--entity` フラグで指定します。

```bash
 wandb job list --entity your-entity --project project-name
```

## ジョブのステータスを確認する

次の表は、キューに入れられた run が持つことができるステータスを定義しています。

| Status | Description |
| --- | --- |
| **Idle** | run はアクティブなエージェントのないキューにあります。 |
| **Queued** | run はエージェントが処理するのを待機しているキューにあります。 |
| **Pending** | run はエージェントによって取得されましたが、まだ開始されていません。これは、 cluster でリソースが利用できないことが原因である可能性があります。 |
| **Running** | run は現在実行中です。 |
| **Killed** | ジョブは user によって強制終了されました。 |
| **Crashed** | run はデータの送信を停止したか、正常に開始されませんでした。 |
| **Failed** | run がゼロ以外の終了コードで終了したか、run の開始に失敗しました。 |
| **Finished** | ジョブは正常に完了しました。 |
