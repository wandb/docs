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

W&B Appを使って対話的に、またはW&B CLIを使ってプログラム的にジョブをキューに追加します。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'W&B CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&B Appを使ってプログラム的にジョブをキューに追加します。

1. W&B Projectページに移動します。
2. 左側のパネルで**Jobs**アイコンを選択します:
  ![](/images/launch/project_jobs_tab_gs.png)
3. **Jobs**ページには、以前に実行されたW&B Runsから作成されたW&B Launchジョブのリストが表示されます。
  ![](/images/launch/view_jobs.png)
4. ジョブ名の横にある**Launch**ボタンを選択します。ページの右側にモーダルが表示されます。
5. **Job version**ドロップダウンから、使用したいLaunchジョブのバージョンを選択します。Launchジョブは他の[W&B Artifact](../artifacts/create-a-new-artifact-version.md)と同様にバージョン管理されます。ソフトウェア依存関係やジョブを実行するためのソースコードを変更すると、同じLaunchジョブの異なるバージョンが作成されます。
6. **Overrides**セクション内で、Launchジョブに設定されている入力に対して新しい値を提供します。一般的なオーバーライドには、新しいエントリーポイントコマンド、引数、または新しいW&B Runの`wandb.config`内の値が含まれます。
  ![](/images/launch/create_starter_queue_gs.png)
  **Paste from...**ボタンをクリックすると、他のW&B Runsから使用したLaunchジョブの値をコピー＆ペーストできます。
7. **Queue**ドロップダウンから、Launchジョブを追加したいLaunchキューの名前を選択します。
8. **Job Priority**ドロップダウンを使用して、Launchジョブの優先順位を指定します。Launchキューが優先順位をサポートしていない場合、Launchジョブの優先順位は「Medium」に設定されます。
9. **(オプション) このステップは、チーム管理者がキュー設定テンプレートを作成した場合にのみ実行してください**  
**Queue Configurations**フィールド内で、チームの管理者が作成した設定オプションの値を提供します。  
例えば、以下の例では、チーム管理者がチームが使用できるAWSインスタンスタイプを設定しています。この場合、チームメンバーは`ml.m4.xlarge`または`ml.p3.xlarge`のコンピュートインスタンスタイプを選択してモデルをトレーニングできます。
![](/images/launch/team_member_use_config_template.png)
10. **Destination project**を選択します。結果のRunが表示されるプロジェクトです。このプロジェクトはキューと同じエンティティに属している必要があります。
11. **Launch now**ボタンを選択します。

  </TabItem>
  <TabItem value="cli">

`wandb launch`コマンドを使用してジョブをキューに追加します。ハイパーパラメーターのオーバーライドを含むJSON設定を作成します。例えば、[Quickstart](./walkthrough.md)ガイドのスクリプトを使用して、以下のオーバーライドを含むJSONファイルを作成します:

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

キュー設定をオーバーライドしたい場合、またはLaunchキューに設定リソースが定義されていない場合、`config.json`ファイル内で`resource_args`キーを指定できます。例えば、上記の例を続けると、`config.json`ファイルは次のようになります:

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
W&B Teamで作業している場合、`entity`フラグ（`-e`）を指定して、キューが使用するエンティティを示すことをお勧めします。

  </TabItem>
</Tabs>