---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Enqueue jobs

以下のページでは、launch queueにジョブを追加する方法について説明します。

:::info
あなたやチームの誰かが既にlaunch queueを設定していることを確認してください。詳細は、[Set up Launch](./setup-launch.md)ページを参照してください。
:::

## キューにジョブを追加する

W&B Appを使用して対話的に、またはW&B CLIを使用してプログラム的にキューにジョブを追加できます。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'W&B CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&B Appを使用してプログラム的にキューにジョブを追加します。

1. W&B Projectページに移動します。
2. 左パネルの**Jobs**アイコンを選択します:
  ![](/images/launch/project_jobs_tab_gs.png)
3. **Jobs**ページには、以前に実行されたW&B runsから作成されたW&B launch jobsのリストが表示されます。
  ![](/images/launch/view_jobs.png)
4. ジョブ名の横にある**Launch**ボタンを選択します。ページの右側にモーダルが表示されます。
5. **Job version**のドロップダウンから、使用したいlaunch jobのバージョンを選択します。Launch jobsは他の[W&B Artifact](../artifacts/create-a-new-artifact-version.md)同様にバージョン管理されます。ソフトウェアの依存関係やジョブを実行するためのソースコードが変更されると、同じlaunch jobの異なるバージョンが作成されます。
6. **Overrides**セクションで、launch jobに設定されている入力値の新しい値を提供します。共通のオーバーライドには、新しいエントリーポイントコマンド、引数、または新しいW&B runの`wandb.config`の値が含まれます。
  ![](/images/launch/create_starter_queue_gs.png)
  **Paste from...**ボタンをクリックすることで、他のW&B runsから使用されたlaunch jobの値をコピー＆ペーストできます。
7. **Queue**のドロップダウンから、launch jobを追加したいlaunch queueの名前を選択します。
8. **Job Priority**のドロップダウンを使用して、launch jobの優先度を指定します。Launch jobの優先度は、launch queueが優先度設定をサポートしていない場合、「Medium」に設定されます。
9. **（オプション）この手順は、チーム管理者がキュー設定テンプレートを作成した場合のみ実行してください**  
 **Queue Configurations**フィールドで、チーム管理者が作成した設定オプションの値を提供します。  
たとえば、以下の例では、チーム管理者がチームで使用できるAWSインスタンスタイプを設定しています。この場合、チームメンバーは`ml.m4.xlarge`または`ml.p3.xlarge`のコンピュートインスタンスタイプを選択してモデルをトレーニングできます。
![](/images/launch/team_member_use_config_template.png)
10. **Destination project**を選択します。ここに結果のrunが表示されます。このプロジェクトはキューと同じエンティティに属する必要があります。
11. **Launch now**ボタンを選択します。


  </TabItem>
  <TabItem value="cli">

`wandb launch` コマンドを使用してキューにジョブを追加します。ハイパーパラメータオーバーライドを含むJSON設定を作成します。たとえば、[Quickstart](./walkthrough.md)ガイドのスクリプトを使用して、次のようなオーバーライドを含むJSONファイルを作成します:

```json title="config.json"
{
  "overrides": {
      "args": [],
      "run_config": {
          "learning_rate": 0,
          "epochs": 0
      },
      "entry_point": []
  }
}
```

:::note
JSON設定ファイルを提供しない場合、W&B Launchはデフォルトのパラメータを使用します。
:::

キュー設定をオーバーライドするか、またはlaunch queueに設定リソースが定義されていない場合、設定ファイルの中で `resource_args` キーを指定できます。例えば、上記の例を続けると、設定ファイルは次のようになります:

```json title="config.json"
{
  "overrides": {
      "args": [],
      "run_config": {
          "learning_rate": 0,
          "epochs": 0
      },
      "entry_point": []
  },
  "resource_args": {
        "<resource-type>" : {
            "<key>": "<value>"
        }
  }
}
```

`<>` 内の値を自分の値に置き換えます。

`queue`(`-q`) フラグにキューの名前を、 `job`(`-j`) フラグにジョブの名前を、 `config`(`-c`) フラグに設定ファイルのパスを指定します。

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B Teamで作業する場合、キューが使用するエンティティを示すために `entity` フラグ (`-e`) を指定することをお勧めします。


  </TabItem>
</Tabs>