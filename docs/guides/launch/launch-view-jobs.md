---
displayed_sidebar: default
---

# View launch jobs

このページでは、キューに追加されたlaunch jobsの情報を表示する方法について説明します。

## View jobs

W&Bアプリを使用して、キューに追加されたジョブを表示します。

1. https://wandb.ai/home にアクセスします。
2. 左サイドバーの**Applications**セクション内の**Launch**を選択します。
3. **All entities**のドロップダウンを選択し、launch jobが属するエンティティを選択します。
4. Launch Applicationページから折りたたみ可能なUIを展開し、その特定のキューに追加されたジョブのリストを表示します。

:::info
launch agentがlaunch jobを実行すると、runが作成されます。言い換えれば、リストされている各runは、そのキューに追加された特定のジョブに対応しています。
:::

例えば、以下の画像は`job-source-launch_demo-canonical`というジョブから作成された2つのrunを示しています。このジョブは`Start queue`というキューに追加されました。キュー内で最初にリストされているrunは`resilient-snowball`で、2番目にリストされているrunは`earthy-energy-165`です。

![](/images/launch/launch_jobs_status.png)

W&BアプリのUI内では、launch jobsから作成されたrunに関する追加情報を見つけることができます。例えば:
   - **Run**: そのジョブに割り当てられたW&B runの名前。
   - **Job ID**: ジョブの名前。
   - **Project**: runが属するプロジェクトの名前。
   - **Status**: キューに入れられたrunのステータス。
   - **Author**: runを作成したW&Bエンティティ。
   - **Creation date**: キューが作成されたタイムスタンプ。
   - **Start time**: ジョブが開始されたタイムスタンプ。
   - **Duration**: ジョブのrunを完了するのにかかった時間（秒単位）。

## List jobs 
W&B CLIを使用して、プロジェクト内に存在するジョブのリストを表示します。W&B job listコマンドを使用し、launch jobが属するプロジェクトとエンティティの名前を`--project`と`--entity`フラグにそれぞれ指定します。

```bash
 wandb job list --entity your-entity --project project-name
```

## Check the status of a job

次の表は、キューに入れられたrunが持つ可能性のあるステータスを定義しています：

| Status | Description |
| --- | --- |
| **Idle** | runはアクティブなエージェントがいないキューにあります。 |
| **Queued** | runはエージェントが処理するのを待っているキューにあります。 |
| **Pending** | runはエージェントにピックアップされましたが、まだ開始されていません。これはクラスター上でリソースが利用できないためかもしれません。 |
| **Running** | runは現在実行中です。 |
| **Killed** | ジョブはユーザーによって停止されました。 |
| **Crashed** | runはデータの送信を停止したか、正常に開始されませんでした。 |
| **Failed** | runは非ゼロの終了コードで終了したか、開始に失敗しました。 |
| **Finished** | ジョブは正常に完了しました。 |