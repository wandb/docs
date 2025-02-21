---
title: Add job to queue
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-add-job-to-queue
    parent: create-and-deploy-jobs
url: guides/launch/add-job-to-queue
---

以下のページでは、ローンンチジョブをローンンチキューに追加する方法を説明します。

{{% alert %}}
あなた、またはチームの誰かがすでにローンンチキューを設定済みであることを確認してください。詳細は、[Set up Launch]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

## キューにジョブを追加する

W&B Appを使用してインタラクティブに、またはW&B CLIを使用してプログラマティックにキューにジョブを追加できます。

{{< tabpane text=true >}}
{{% tab "W&B app" %}}
W&B Appを使用してプログラム的にキューにジョブを追加します。

1. W&B プロジェクトページに移動します。
2. 左パネルの**Jobs**アイコンを選択します。
  {{< img src="/images/launch/project_jobs_tab_gs.png" alt="" >}}
3. **Jobs**ページには、以前に実行されたW&B runから作成されたW&Bローンンチジョブのリストが表示されます。
  {{< img src="/images/launch/view_jobs.png" alt="" >}}
4. ジョブ名の横にある**Launch**ボタンを選択します。ページの右側にモーダルが表示されます。
5. **Job version**のドロップダウンから、使用するローンンチジョブのバージョンを選択します。ローンンチジョブは、他の[W&B Artifact]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}})と同様にバージョン管理されます。ソフトウェアの依存関係やジョブの実行に使用するソースコードを変更すると、同じローンンチジョブの異なるバージョンが作成されます。
6. **Overrides**セクション内で、ローンンチジョブに対して設定済みの入力に対して新しい値を提供します。一般的なオーバーライドには、新しいエントリーポイントコマンド、引数、または新しいW&B runの`wandb.config`の値が含まれます。
  {{< img src="/images/launch/create_starter_queue_gs.png" alt="" >}}
  **Paste from...** ボタンをクリックすることで、他のW&B runからローンンチジョブを使用した値をコピーアンドペーストできます。
7. **Queue**のドロップダウンから、ローンンチジョブを追加するローンンチキューの名前を選択します。
8. **Job Priority**のドロップダウンを使用して、ローンンチジョブの優先順位を指定します。ローンンチキューが優先順位をサポートしていない場合、ローンンチジョブの優先順位は "Medium" に設定されます。
9. **(オプション) キュー設定テンプレートがチームの管理者によって作成されている場合のみ、このステップを行ってください**
**Queue Configurations** フィールドに、チームの管理者によって作成された設定オプションの値を入力します。
例えば、以下の例では、チーム管理者はAWSインスタンスタイプを設定し、チームメンバーが`ml.m4.xlarge` または `ml.p3.xlarge` のコンピュートインスタンスタイプを選択してモデルをトレーニングできるようにしました。
{{< img src="/images/launch/team_member_use_config_template.png" alt="" >}}
10. **Destination project**を選択します。このプロジェクトはキューと同じエンティティに属している必要があります。
11. **Launch now**ボタンを選択します。

{{% /tab %}}
{{% tab "W&B CLI" %}}

`wandb launch` コマンドを使用してキューにジョブを追加します。ハイパーパラメーターオーバーライド付きのJSON設定を作成します。例えば、[クイックスタート]({{< relref path="../walkthrough.md" lang="ja" >}})ガイドのスクリプトを使用すると、以下のオーバーライドを含むJSONファイルを作成します。

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

{{% alert %}}
JSON設定ファイルを提供しない場合、W&B Launchはデフォルトのパラメータを使用します。
{{% /alert %}}

キュー設定を上書きしたい場合、またはローンンチキューに設定リソースが定義されていない場合は、config.jsonファイルで`resource_args`キーを指定できます。例えば、上記の例を続けると、config.jsonファイルは次のようになるかもしれません。

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

`<>`内の値をあなた自身の値に置き換えてください。

`queue`(`-q`) フラグのキュー名、`job`(`-j`) フラグのジョブ名、および `config`(`-c`) フラグの設定ファイルへのパスを提供してください。

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B チームで作業する場合、`entity`フラグ(`-e`)を指定して、キューが使用するエンティティを示すことをお勧めします。

{{% /tab %}}
{{% /tabpane %}}