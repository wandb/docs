---
title: キューにジョブを追加
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-add-job-to-queue
    parent: create-and-deploy-jobs
url: /ja/guides/launch/add-job-to-queue
---

次のページでは、ローンチキューにローンチジョブを追加する方法について説明しています。

{{% alert %}}
あなた、またはチームの誰かが既にローンチキューを設定していることを確認してください。詳細については、[Set up Launch]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

## キューにジョブを追加する

W&B Appを使用してインタラクティブに、またはW&B CLIを使用してプログラム的にキューにジョブを追加します。

{{< tabpane text=true >}}
{{% tab "W&B App" %}}
W&B Appを使用してプログラム的にキューにジョブを追加します。

1. W&B Project Pageに移動します。
2. 左のパネルで **Jobs** アイコンを選択します:
  {{< img src="/images/launch/project_jobs_tab_gs.png" alt="" >}}
3. **Jobs** ページには、以前に実行されたW&B runsから作成されたW&Bローンチジョブのリストが表示されます。 
  {{< img src="/images/launch/view_jobs.png" alt="" >}}
4. ジョブ名の横にある **Launch** ボタンを選択します。ページの右側にモーダルが表示されます。
5. **Job version** ドロップダウンから使用するローンチジョブのバージョンを選択します。ローンチジョブは他の [W&B Artifact]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}}) と同様にバージョン管理されます。ソフトウェアの依存関係やジョブを実行するために使用されるソースコードに変更を加えると、同じローンチジョブの異なるバージョンが作成されます。
6. **Overrides** セクションで、ローンチジョブに設定された入力の新しい値を提供します。一般的なオーバーライドには、新しいエントリーポイントコマンド、引数、または新しいW&B runの `wandb.config` 内の値が含まれます。  
  {{< img src="/images/launch/create_starter_queue_gs.png" alt="" >}}
  **Paste from...** ボタンをクリックして、ローンチジョブで使用された他のW&B runsから値をコピーして貼り付けることができます。
7. **Queue** ドロップダウンから、ローンチジョブを追加するローンチキューの名前を選択します。
8. **Job Priority** ドロップダウンを使用して、ローンチジョブの優先度を指定します。ローンチキューが優先度をサポートしていない場合は、ローンチジョブの優先度は「Medium」に設定されます。
9. **(オプション) この手順は、キュー設定テンプレートがチーム管理者によって作成されている場合にのみ従ってください**  
   **Queue Configurations** フィールド内で、チームの管理者によって作成された設定オプションに対する値を提供します。  
   例えば、次の例では、チーム管理者がチームが使用できるAWSインスタンスタイプを設定しました。この場合、チームメンバーは `ml.m4.xlarge` または `ml.p3.xlarge` のいずれかのコンピュートインスタンスタイプを選択してモデルをトレーニングできます。
{{< img src="/images/launch/team_member_use_config_template.png" alt="" >}}
10. **Destination project** を選択して、結果として生成されるrunが表示されるプロジェクトを指定します。このプロジェクトは、キューと同じエンティティに属している必要があります。
11. **Launch now** ボタンを選択します。 

{{% /tab %}}
{{% tab "W&B CLI" %}}

`wandb launch` コマンドを使用して、キューにジョブを追加します。ハイパーパラメーターオーバーライドを含むJSON設定を作成します。例えば、[クイックスタート]({{< relref path="../walkthrough.md" lang="ja" >}}) ガイドのスクリプトを使用して、以下のオーバーライドを含むJSONファイルを作成します。

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
JSON設定ファイルを提供しない場合、W&B Launchはデフォルトのパラメーターを使用します。
{{% /alert %}}

キューの設定をオーバーライドする場合、またはローンチキューに設定リソースが定義されていない場合、`config.json` ファイルで `resource_args` キーを指定できます。例えば、上記の例を続けると、`config.json` ファイルは次のようになります。

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

`<>` 内の値を独自の値に置き換えてください。

`queue`（`-q`）フラグにはキューの名前を、`job`（`-j`）フラグにはジョブの名前を、`config`（`-c`）フラグには設定ファイルのパスを指定してください。

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B Team内で作業する場合は、`entity` フラグ（`-e`）を指定して、キューが使用するエンティティを示すことをお勧めします。

{{% /tab %}}
{{% /tabpane %}}