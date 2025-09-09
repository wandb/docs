---
title: ジョブをキューに追加
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-add-job-to-queue
    parent: create-and-deploy-jobs
url: guides/launch/add-job-to-queue
---

このページでは、Launch キューに Launch ジョブを追加する方法を説明します。
{{% alert %}}
あなた、またはチームの誰かが、すでに Launch キューを設定済みであることを確認してください。詳しくは、[Launch のセットアップ]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

## キューにジョブを追加する

W&B App を使って対話的に、または W&B CLI を使ってプログラムから、キューにジョブを追加できます。

{{< tabpane text=true >}}
{{% tab "W&B App" %}}
W&B App を使って、キューにジョブをプログラムから追加します。

1. W&B の Project ページに移動します。
2. 左側の パネル で **Jobs** アイコンを選択します:
  {{< img src="/images/launch/project_jobs_tab_gs.png" alt="Project の Jobs タブ" >}}
3. **Jobs** ページには、以前に実行された W&B の run から作成された W&B の Launch ジョブの一覧が表示されます。 
  {{< img src="/images/launch/view_jobs.png" alt="Jobs の一覧" >}}
4. ジョブ名の横にある **Launch** ボタンを選択します。ページ右側にモーダルが表示されます。
5. **Job version** ドロップダウンから、使用したい Launch ジョブの バージョン を選択します。Launch ジョブは、他の [W&B Artifact]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}}) と同様に バージョン管理 されています。同じ Launch ジョブでも、ジョブの実行に使用するソフトウェア依存関係やソース コードを変更すると、別の バージョン が作成されます。
6. **Overrides** セクションで、Launch ジョブに設定された入力に対して新しい 値 を指定します。一般的なオーバーライドには、新しいエントリポイント コマンド、引数、または新しい W&B run の `wandb.Run.config` 内の 値 などがあります。  
  {{< img src="/images/launch/create_starter_queue_gs.png" alt="キューの設定" >}}
  **Paste from...** ボタンをクリックすると、この Launch ジョブを使用した他の W&B run から 値 をコピー＆ペーストできます。
7. **Queue** ドロップダウンから、この Launch ジョブを追加したい Launch キュー名を選択します。 
8. **Job Priority** ドロップダウンで、この Launch ジョブの優先度を指定します。Launch キューが優先度をサポートしていない場合、Launch ジョブの優先度は "Medium" に設定されます。
9. **(任意) キューの config テンプレートがチーム管理者によって作成されている場合のみ実施**  
**Queue Configurations** フィールドで、チームの管理者が作成した設定オプションの 値 を指定します。  
例として、以下ではチーム管理者がチームで使用できる AWS のインスタンスタイプを設定しています。この場合、チーム メンバーは学習用のコンピュート インスタンスとして `ml.m4.xlarge` または `ml.p3.xlarge` を選択できます。
{{< img src="/images/launch/team_member_use_config_template.png" alt="Config テンプレートの選択" >}}
10. 結果の run が表示される **Destination project** を選択します。この Project は、キューと同じ Entity に属している必要があります。
11. **Launch now** ボタンを選択します。 

{{% /tab %}}
{{% tab "W&B CLI" %}}

`wandb launch` コマンドを使って、ジョブをキューに追加します。ハイパーパラメーターのオーバーライドを含む JSON 設定を作成します。例えば、[クイックスタート]({{< relref path="../walkthrough.md" lang="ja" >}}) ガイドの スクリプト を使う場合、次のオーバーライドを含む JSON ファイルを作成します:

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
JSON 設定ファイルを指定しない場合、W&B Launch はデフォルトの パラメータ を使用します。
{{% /alert %}}

キューの 設定 を上書きしたい場合、または Launch キューに設定リソースが定義されていない場合は、config.json ファイルで `resource_args` キーを指定できます。例えば、上の例の続きとして、config.json ファイルは次のようになります:

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

`<>` 内の 値 はご自身の 値 に置き換えてください。

`queue`（`-q`）フラグにはキュー名を、`job`（`-j`）フラグにはジョブ名を、`config`（`-c`）フラグには設定ファイルのパスを指定します。

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B の Team で作業している場合は、キューが使用する Entity を示すために `entity` フラグ（`-e`）を指定することをお勧めします。

{{% /tab %}}
{{% /tabpane %}}