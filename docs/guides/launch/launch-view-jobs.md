---
displayed_sidebar: default
---

# Launchジョブの表示

このページでは、キューに追加されたLaunchジョブの情報を表示する方法について説明します。

## ジョブの表示

W&Bアプリを使用してキューに追加されたジョブを表示します。

1. https://wandb.ai/home にアクセスします。
2. 左サイドバーの**Applications**セクション内の**Launch**を選択します。
3. **All entities**のドロップダウンを選択し、Launchジョブが属するエンティティを選択します。
4. Launch Applicationページから折りたたみ可能なUIを展開して、その特定のキューに追加されたジョブのリストを表示します。

:::info
LaunchエージェントがLaunchジョブを実行すると、runが作成されます。言い換えれば、リストされている各runは、そのキューに追加された特定のジョブに対応しています。
:::

例えば、以下の画像は`job-source-launch_demo-canonical`というジョブから作成された2つのrunを示しています。このジョブは`Start queue`というキューに追加されました。キュー内で最初にリストされているrunは`resilient-snowball`で、2番目にリストされているrunは`earthy-energy-165`です。

![](/images/launch/launch_jobs_status.png)

W&BアプリのUI内では、Launchジョブから作成されたrunに関する追加情報を見つけることができます。例えば：
   - **Run**: そのジョブに割り当てられたW&B runの名前。
   - **Job ID**: ジョブの名前。
   - **Project**: runが属するプロジェクトの名前。
   - **Status**: キューに入れられたrunのステータス。
   - **Author**: runを作成したW&Bエンティティ。
   - **Creation date**: キューが作成されたタイムスタンプ。
   - **Start time**: ジョブが開始されたタイムスタンプ。
   - **Duration**: ジョブのrunを完了するのにかかった時間（秒単位）。

## ジョブのリスト表示
W&B CLIを使用して、プロジェクト内に存在するジョブのリストを表示します。W&B job listコマンドを使用し、Launchジョブが属するプロジェクトとエンティティの名前をそれぞれ`--project`と`--entity`フラグで指定します。

```bash
 wandb job list --entity your-entity --project project-name
```

## ジョブのステータス確認

以下の表は、キューに入れられたrunが持つ可能性のあるステータスを定義しています：

| ステータス | 説明 |
| --- | --- |
| **Idle** | runはアクティブなエージェントのないキューにあります。 |
| **Queued** | runはエージェントが処理するのを待っているキューにあります。 |
| **Pending** | runはエージェントにピックアップされましたが、まだ開始されていません。これはクラスター上のリソースが利用できないためかもしれません。 |
| **Running** | runは現在実行中です。 |
| **Killed** | ジョブはユーザーによって終了されました。 |
| **Crashed** | runはデータの送信を停止したか、正常に開始されませんでした。 |
| **Failed** | runは非ゼロの終了コードで終了したか、runの開始に失敗しました。 |
| **Finished** | ジョブは正常に完了しました。 |