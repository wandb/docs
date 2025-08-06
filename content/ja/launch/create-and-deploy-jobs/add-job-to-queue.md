---
title: ジョブをキューに追加
menu:
  launch:
    identifier: add-job-to-queue
    parent: create-and-deploy-jobs
url: guides/launch/add-job-to-queue
---

このページでは、ローンチジョブをローンチキューに追加する方法について説明します。

{{% alert %}}
あなた、またはチームの誰かが、すでにローンチキューを設定していることを確認してください。詳細は [Set up Launch]({{< relref "/launch/set-up-launch/" >}}) ページをご覧ください。
{{% /alert %}}

## キューへのジョブ追加

W&B App を使ってインタラクティブに、または W&B CLI を使ってプログラム的にキューへジョブを追加できます。

{{< tabpane text=true >}}
{{% tab "W&B app" %}}
W&B App を使ってプログラム的にジョブをキューへ追加できます。

1. W&B Project ページへ移動します。
2. 左側のパネルで **Jobs** アイコンを選択します:
  {{< img src="/images/launch/project_jobs_tab_gs.png" alt="Project Jobs tab" >}}
3. **Jobs** ページでは、これまでに実行された W&B Run から作成された W&B ローンチジョブの一覧が表示されます。
  {{< img src="/images/launch/view_jobs.png" alt="Jobs listing" >}}
4. ジョブ名の横にある **Launch** ボタンを選択します。ページ右側にモーダルが表示されます。
5. **Job version** のドロップダウンから、使用したいローンチジョブのバージョンを選択します。ローンチジョブも他の [W&B Artifact]({{< relref "/guides/core/artifacts/create-a-new-artifact-version.md" >}}) と同様にバージョン管理されます。ソフトウェア依存関係やジョブの実行に使うソースコードを変更した場合、同じローンチジョブでも異なるバージョンが作成されます。
6. **Overrides** セクションで、ローンチジョブに設定された各入力値に対して新しい値を入力します。よくあるオーバーライドとしては、新しいエントリポイントコマンドや引数、あるいは新しい W&B Run の `wandb.Run.config` の値などがあります。  
  {{< img src="/images/launch/create_starter_queue_gs.png" alt="Queue configuration" >}}
  **Paste from...** ボタンをクリックすることで、他の W&B Run で使われた設定値をコピー＆ペーストすることも可能です。
7. **Queue** のドロップダウンから、追加したいローンチキュー名を選択します。
8. **Job Priority** ドロップダウンで、ローンチジョブの優先度を選択します。ローンチキューが優先度をサポートしていない場合、優先度は「Medium」に設定されます。
9. **（オプション: チーム管理者がキュー設定テンプレートを作成済みの場合のみこの手順を実施してください）**  
**Queue Configurations** フィールドで、チームの管理者が作成した設定オプションに値を入力します。  
たとえば、以下の例ではチーム管理者が使用可能な AWS インスタンスタイプを設定しています。この場合、チームメンバーは `ml.m4.xlarge` または `ml.p3.xlarge` のいずれかのインスタンスタイプを選択してモデルのトレーニングに使うことができます。
{{< img src="/images/launch/team_member_use_config_template.png" alt="Config template selection" >}}
10. 結果として生成される Run を表示したい **Destination project** を選択します。このプロジェクトは、キューと同じ Entity に属している必要があります。
11. **Launch now** ボタンを選択します。

{{% /tab %}}
{{% tab "W&B CLI" %}}

`wandb launch` コマンドでキューへジョブを追加できます。ハイパーパラメーターのオーバーライドを含む JSON 設定ファイルを作成します。例えば、[Quickstart]({{< relref "../walkthrough.md" >}}) ガイドのスクリプトを使う場合、以下のようなオーバーライドを含んだ JSON ファイルを作成します。

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
JSON 設定ファイルを指定しない場合、W&B Launch はデフォルトのパラメータを使用します。
{{% /alert %}}

キュー設定をオーバーライドしたい場合や、ローンチキューに設定リソースが定義されていない場合は、config.json ファイルの中で `resource_args` キーを指定できます。たとえば、上記の例から続けて、config.json ファイルは次のように記述できます。

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

`<>` 内の値はあなた自身の値に置き換えてください。

`queue`（`-q`）フラグにはキューの名前、`job`（`-j`）フラグにはジョブの名前、`config`（`-c`）フラグには設定ファイルへのパスを指定します。

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B Team 内で作業している場合は、どの Entity のキューを使うか明示するため、`entity` フラグ（`-e`）の指定を推奨します。

{{% /tab %}}
{{% /tabpane %}}