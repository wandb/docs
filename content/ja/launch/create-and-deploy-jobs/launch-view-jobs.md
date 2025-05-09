---
title: ローンンチ ジョブを表示する
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-launch-view-jobs
    parent: create-and-deploy-jobs
url: /ja/guides/launch/launch-view-jobs
---

以下のページでは、キューに追加されたローンンチジョブの情報を表示する方法を説明します。

## ジョブの表示

W&B アプリケーションを使用してキューに追加されたジョブを表示します。

1. https://wandb.ai/home にある W&B アプリケーションにアクセスします。
2. 左側のサイドバーにある **Applications** セクション内の **Launch** を選択します。
3. **All entities** ドロップダウンを選択し、ローンンチジョブが所属するエンティティを選択します。
4. Launch Application ページから折りたたみ可能なUIを展開し、その特定のキューに追加されたジョブのリストを表示します。

{{% alert %}}
ローンンチエージェントがローンンチジョブを実行すると、run が作成されます。つまり、リストされている各runは、そのキューに追加された特定のジョブに対応しています。
{{% /alert %}}

例えば、次の画像は、`job-source-launch_demo-canonical`というジョブから作成された2つのrunを示しています。このジョブは `Start queue` というキューに追加されました。キューにリストされている最初のrunは `resilient-snowball` と呼ばれ、2番目のrunは `earthy-energy-165` と呼ばれます。

{{< img src="/images/launch/launch_jobs_status.png" alt="" >}}

W&B アプリケーションUI内で、ローンンチジョブから作成されたrunに関する次のような追加情報を見つけることができます：
   - **Run**: そのジョブに割り当てられた W&B run の名前。
   - **Job ID**: ジョブの名前。
   - **Project**: runが所属するプロジェクトの名前。
   - **Status**: キューに入れられたrunのステータス。
   - **Author**: run を作成した W&B エンティティ。
   - **Creation date**: キューが作成されたタイムスタンプ。
   - **Start time**: ジョブが開始されたタイムスタンプ。
   - **Duration**: ジョブのrunを完了するのにかかった時間（秒単位）。

## ジョブのリスト 
プロジェクト内に存在するジョブのリストを W&B CLI を使用して表示します。W&B job list コマンドを使用し、`--project` および `--entity` フラグにローンンチジョブが所属するプロジェクト名とエンティティ名を指定します。

```bash
wandb job list --entity your-entity --project project-name
```

## ジョブのステータスを確認する

次の表は、キューに入れられたrunが持つ可能性のあるステータスを定義しています：

| ステータス | 説明 |
| --- | --- |
| **Idle** | runはアクティブなエージェントのないキューにあります。 |
| **Queued** | runはエージェントが処理するのを待っているキューにあります。 |
| **Pending** | run はエージェントによって取得されましたが、まだ開始されていません。これはクラスターでリソースが利用できないことが原因である可能性があります。 |
| **Running** | run は現在実行中です。 |
| **Killed** | ジョブはユーザーによって終了されました。 |
| **Crashed** | run はデータの送信を停止したか、正常に開始しませんでした。 |
| **Failed** | run は非ゼロの終了コードで終了したか、run の開始に失敗しました。 |
| **Finished** | ジョブは正常に完了しました。 |