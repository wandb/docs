---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ジョブのキューへの追加

このページでは、LaunchジョブをLaunchキューに追加する方法について説明します。

:::info
あなた、またはチームの誰かが既にLaunchキューを設定していることを確認してください。詳細については、[Set up Launch](./setup-launch.md)ページを参照してください。
:::

## キューにジョブを追加する

W&Bアプリを使ってインタラクティブに、またはW&B CLIを使ってプログラム的にキューにジョブを追加します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'W&B CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&Bアプリを使ってプログラム的にキューにジョブを追加します。

1. W&Bプロジェクトページに移動します。
2. 左側のパネルで**Jobs**アイコンを選択します:
  ![](/images/launch/project_jobs_tab_gs.png)
3. **Jobs**ページには、以前に実行されたW&B Runsから作成されたW&B Launchジョブのリストが表示されます。
  ![](/images/launch/view_jobs.png)
4. ジョブ名の横にある**Launch**ボタンを選択します。ページの右側にモーダルが表示されます。
5. **Job version**ドロップダウンから、使用したいLaunchジョブのバージョンを選択します。Launchジョブは他の[W&B Artifact](../artifacts/create-a-new-artifact-version.md)と同様にバージョン管理されています。ソフトウェア依存関係やジョブを実行するためのソースコードに変更を加えると、同じLaunchジョブの異なるバージョンが作成されます。
6. **Overrides**セクション内で、Launchジョブに設定されている入力に対して新しい値を提供します。一般的なオーバーライドには、新しいエントリーポイントコマンド、引数、または新しいW&B Runの`wandb.config`内の値が含まれます。
  ![](/images/launch/create_starter_queue_gs.png)
  **Paste from...**ボタンをクリックすることで、他のW&B Runsから使用したLaunchジョブの値をコピーして貼り付けることができます。
7. **Queue**ドロップダウンから、Launchジョブを追加したいLaunchキューの名前を選択します。
8. **Job Priority**ドロップダウンを使用して、Launchジョブの優先順位を指定します。Launchキューが優先順位をサポートしていない場合、Launchジョブの優先順位は「Medium」に設定されます。
9. **(オプション) このステップは、チーム管理者がキュー設定テンプレートを作成した場合のみ行います**  
**Queue Configurations**フィールドに、チームの管理者が作成した設定オプションの値を入力します。  
例えば、以下の例では、チーム管理者がチームが使用できるAWSインスタンスタイプを設定しました。この場合、チームメンバーは`ml.m4.xlarge`または`ml.p3.xlarge`のコンピュートインスタンスタイプを選択してモデルをトレーニングできます。
![](/images/launch/team_member_use_config_template.png)
10. 結果のRunが表示される**Destination project**を選択します。このプロジェクトはキューと同じエンティティに属している必要があります。
11. **Launch now**ボタンを選択します。

  </TabItem>
    <TabItem value="cli">

`wandb launch`コマンドを使用してキューにジョブを追加します。ハイパーパラメーターのオーバーライドを含むJSON設定を作成します。例えば、[Quickstart](./walkthrough.md)ガイドのスクリプトを使用して、以下のオーバーライドを含むJSONファイルを作成します:

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
JSON設定ファイルを提供しない場合、W&B Launchはデフォルトのパラメーターを使用します。
:::

キュー設定をオーバーライドしたい場合、またはLaunchキューに設定リソースが定義されていない場合、`resource_args`キーをconfig.jsonファイルに指定できます。例えば、上記の例を続けると、config.jsonファイルは次のようになります:

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

`<>`内の値を自分の値に置き換えてください。

`queue`(`-q`)フラグにはキューの名前を、`job`(`-j`)フラグにはジョブの名前を、`config`(`-c`)フラグには設定ファイルのパスを指定します。

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&Bチーム内で作業している場合、キューが使用するエンティティを示すために`entity`フラグ(`-e`)を指定することをお勧めします。

  </TabItem>
</Tabs>