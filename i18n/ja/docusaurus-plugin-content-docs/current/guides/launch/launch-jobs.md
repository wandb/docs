---
description: W&Bキューにジョブを追加する方法を学びましょう。
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ランを起動する

W&B launchを使って、実行用のキューにジョブを追加して、スイープエージェントによって実行できるようにします。次のガイドでは、キューにランを送信する方法を説明します。

## キューにジョブを追加する
W&Bアプリを使ってインタラクティブにキューにジョブを追加するか、CLIを使ってプログラム的に追加します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&Bアプリ', value: 'app'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&Bアプリを使ってキューにジョブを追加します。

1. W&Bプロジェクトページに移動します。
2. 左パネルの**Jobs**アイコンを選択します:

![](/images/launch/project_jobs_tab_gs.png)

3. **Jobs**ページには、以前に実行されたW&Bランから作成されたW&Bジョブのリストが表示されます。

![](/images/launch/view_jobs.png)

4. ジョブ名の横にある**起動**ボタンを選択します。ページの右側にモーダルが表示されます。
5. そのモーダルから以下を選択してください。
    * **Job version**のドロップダウンから、キューに追加したいJob versionを選択してください。この例では、バージョンが1つしかないため、`v0`を選択します。
    * **Paste from…** ボタンを選択して、特定のW&B Runから使用されたハイパーパラメーターを自動的に伝播させます。以下の画像では、選ぶことができる2つのrunがあります。

![](/images/launch/create_starter_queue_gs.png)

6. 次に、**Queue**のドロップダウンから**Starter queue**を選択して、キューを作成します。
7. **Launch now**ボタンを選択します。


  </TabItem>
    <TabItem value="cli">

`wandb launch`コマンドを使ってキューにジョブを追加します。ハイパーパラメーターのオーバーライド情報を持つJSON設定を作成してください。例えば、[Quickstart](./quickstart.md)ガイドで紹介されているスクリプトを使って、次のようなオーバーライド情報を持つJSONファイルを作成します：

```json
// config.json
{
    "args": [],
    "run_config": {
        "learning_rate": 0,
        "epochs": 0
    },
    "entry_point": []
}

W&B Launchは、JSON設定ファイルを提供しない場合、デフォルトのパラメーターが使用されます。
`queue`（`-q`）フラグにはキューの名前、`job`（`-j`）フラグにはジョブの名前、`config`（`-c`）フラグには設定ファイルへのパスを指定してください。

```bash
wandb launch -j <job> -q <queue-name> -e <entity-name> -c path/to/config.json
```

W&Bチーム内で作業している場合は、キューが使用するエンティティを示すために、`entity`フラグ（`-e`）を指定することをお勧めします。

  </TabItem>
</Tabs>

## キューに追加されたジョブを表示する
W&Bアプリでキューに追加されたジョブを表示します。

1. https://wandb.ai/home のW&Bアプリに移動します。
2. 左側のサイドバーの**Applications**セクションで**Launch**を選択します。
3. **全エンティティ**のドロップダウンを選択し、フィルタリングするエンティティを選択します。
4. Launch Applicationページから折りたたみ可能なキューUIを展開して、特定のキューに追加されたジョブを表示します。

リストに表示されている各runは、そのキューに追加されたジョブに対応しています。例えば、次の画像では、`Starter queue`というキューに2つのジョブがリストされています。1つは`resilient-snowball`と呼ばれ、もう1つは`earthy-energy-165`と呼ばれています。

![](/images/launch/launch_jobs_status.png)

ジョブに関する追加情報を次のように調べます。
   - **Run**：そのジョブに割り当てられたW&B Runの名前。
   - **Job ID**：ジョブの名前。デフォルトの命名に関する情報は、[ジョブ命名規則](create-job#job-naming-conventions)ページを参照してください。
   - **Project**：runが所属するプロジェクトの名前。
   - **Status**：キューに入っているrunのステータス。
   - **Author**：runを作成したW&Bエンティティ。
   - **Creation date**：キューが作成された日時。
   - **Start time**：ジョブが開始された日時。
   - **Duration**：ジョブのrunを完了するまでにかかった時間（秒）。
## キューにあるrunsの状態



| 状態 | 説明 |

| --- | --- |

| **アイドル** | runがアクティブなエージェントのないキューにあります。 |

| **キュー待ち** | エージェントが処理を待っているキューにrunがあります。 |

| **開始中** | エージェントがrunを取得しましたが、まだ開始されていません。 |

| **実行中** | runが現在実行中です。 |

| **強制終了** | ユーザーがジョブを強制終了しました。 |

| **クラッシュ** | runがデータの送信を停止したか、正常に開始されませんでした。 |

| **失敗** | runが0以外の終了コードで終了しました。 |

| **完了** | ジョブが正常に完了しました。 |