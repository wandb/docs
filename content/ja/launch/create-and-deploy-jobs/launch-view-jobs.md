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

W&B アプリケーション でキューに追加されたジョブを表示します。

1. W&B アプリケーション (https://wandb.ai/home) に移動します。
2. 左側のサイドバーの [**Applications**] セクションで [**Launch**] を選択します。
3. [**All entities**] ドロップダウンを選択し、Launch ジョブが属する entity を選択します。
4. Launch アプリケーション ページから、折りたたみ可能な UI を展開して、その特定のキューに追加されたジョブのリストを表示します。

{{% alert %}}
Launch エージェント が Launch ジョブを実行すると、run が作成されます。つまり、リストされている各 run は、そのキューに追加された特定のジョブに対応します。
{{% /alert %}}

たとえば、次の図は、`job-source-launch_demo-canonical` というジョブから作成された 2 つの run を示しています。ジョブは `Start queue` というキューに追加されました。キューにリストされている最初の run は `resilient-snowball` と呼ばれ、2 番目にリストされている run は `earthy-energy-165` と呼ばれます。

{{< img src="/images/launch/launch_jobs_status.png" alt="" >}}

W&B アプリケーション UI 内では、Launch ジョブから作成された run に関する追加情報を見つけることができます。以下に例を示します。
   - **Run**: そのジョブに割り当てられた W&B run の名前。
   - **Job ID**: ジョブの名前。
   - **Project**: run が属する project の名前。
   - **Status**: キューに入れられた run のステータス。
   - **Author**: run を作成した W&B entity 。
   - **Creation date**: キューが作成されたときのタイムスタンプ。
   - **Start time**: ジョブが開始されたときのタイムスタンプ。
   - **Duration**: ジョブの run の完了にかかった時間 (秒単位)。

## ジョブのリスト

W&B CLI で project 内に存在するジョブのリストを表示します。W&B job list コマンドを使用し、Launch ジョブが属する project と entity の名前を、それぞれ `--project` フラグと `--entity` フラグに指定します。

```bash
 wandb job list --entity your-entity --project project-name
```

## ジョブのステータスを確認する

次の表は、キューに入れられた run が持つことができるステータスを定義しています。

| Status | 説明 |
| --- | --- |
| **Idle** | run は、アクティブなエージェント がないキューにあります。 |
| **Queued** | run は、エージェント が処理するのを待機しているキューにあります。 |
| **Pending** | run は エージェント によってピックアップされましたが、まだ開始されていません。これは、 クラスター でリソースが利用できないことが原因である可能性があります。 |
| **Running** | run は現在実行中です。 |
| **Killed** | ジョブは ユーザー によって強制終了されました。 |
| **Crashed** | run はデータの送信を停止したか、正常に開始されませんでした。 |
| **Failed** | run が 0 以外の終了コードで終了したか、run の開始に失敗しました。 |
| **Finished** | ジョブは正常に完了しました。 |
