---
title: View launch jobs
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-launch-view-jobs
    parent: create-and-deploy-jobs
url: guides/launch/launch-view-jobs
---

次のページでは、キューに追加されたローンンチジョブに関する情報を表示する方法を説明します。

## ジョブを表示する

W&B アプリを使用して、キューに追加されたジョブを表示します。

1. https://wandb.ai/home にアクセスして W&B アプリに移動します。
2. 左のサイドバーの **Applications** セクション内の **Launch** を選択します。
3. **All entities** ドロップダウンを選択し、ローンンチジョブに関連するエンティティを選択します。
4. Launch Application ページから折りたたみ UI を展開して、その特定のキューに追加されたジョブのリストを表示します。

{{% alert %}}
ローンンチエージェントがローンンチジョブを実行すると、run が作成されます。言い換えると、リストされた各 run は、そのキューに追加された特定のジョブに対応しています。
{{% /alert %}}

例えば、次の画像は `job-source-launch_demo-canonical` というジョブから作成された2つの run を示しています。このジョブは `Start queue` というキューに追加されました。キュー内で最初にリストされた run は `resilient-snowball` で、2番目にリストされた run は `earthy-energy-165` と呼ばれています。

{{< img src="/images/launch/launch_jobs_status.png" alt="" >}}

W&B アプリ UI 内では、ローンンチジョブから作成された run に関する追加の情報を見つけることができます。以下の情報が含まれます:
   - **Run**: そのジョブに割り当てられた W&B run の名前。
   - **Job ID**: ジョブの名前。
   - **Project**: run が属するプロジェクトの名前。
   - **Status**: キューに置かれた run のステータス。
   - **Author**: run を作成した W&B エンティティ。
   - **Creation date**: キューが作成されたときのタイムスタンプ。
   - **Start time**: ジョブが開始されたときのタイムスタンプ。
   - **Duration**: ジョブの run を完了するのにかかった秒数。

## ジョブをリストする

W&B CLI を使用して、プロジェクト内の存在するジョブのリストを表示します。W&B job list コマンドを使用し、`--project` と `--entity` フラグにローンンチジョブが属するプロジェクト名とエンティティ名を指定します。

```bash
 wandb job list --entity your-entity --project project-name
```

## ジョブのステータスを確認する

次の表は、キューに置かれた run が持つことができるステータスを定義しています:

| Status | 説明 |
| --- | --- |
| **Idle** | run はアクティブなエージェントのないキューにあります。 |
| **Queued** | run はエージェントによって処理されるのを待っているキューにあります。 |
| **Pending** | run はエージェントによって取得されたが、まだ開始されていません。これは、クラスターでリソースが利用可能でないためかもしれません。 |
| **Running** | run は現在実行中です。 |
| **Killed** | ジョブはユーザーによって終了されました。 |
| **Crashed** | run はデータの送信を停止したか、正常に開始されませんでした。 |
| **Failed** | run は非ゼロの終了コードで終了したか、または開始に失敗しました。 |
| **Finished** | ジョブは正常に完了しました。 |