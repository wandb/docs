---
title: キューにジョブを追加
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-add-job-to-queue
    parent: create-and-deploy-jobs
url: guides/launch/add-job-to-queue
---

以下のページでは、ローンチキューに launch ジョブを追加する方法について説明します。

{{% alert %}}
あなた、またはチームメンバーが、すでにローンチキューを設定していることを確認してください。詳しくは[Set up Launch]({{< relref path="/launch/set-up-launch/" lang="ja" >}})ページをご覧ください。
{{% /alert %}}

## キューにジョブを追加する

W&B アプリで対話的に、または W&B CLI を使ってプログラムからキューにジョブを追加できます。

{{< tabpane text=true >}}
{{% tab "W&B app" %}}
W&B アプリから対話的にジョブをキューに追加します。

1. あなたの W&B Project ページに移動します。
2. 左側のパネルから **Jobs** アイコンを選択します。
  {{< img src="/images/launch/project_jobs_tab_gs.png" alt="Project Jobs tab" >}}
3. **Jobs** ページには、過去に実行された W&B Run から作成されたローンチジョブのリストが表示されます。
  {{< img src="/images/launch/view_jobs.png" alt="Jobs listing" >}}
4. ジョブ名の横にある **Launch** ボタンを選択します。ページ右側にモーダルが表示されます。
5. **Job version** ドロップダウンから、使用したいローンチジョブのバージョンを選択します。 Launch ジョブは他の [W&B Artifact]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}}) と同じくバージョン管理されています。ジョブの実行に使うソフトウェア依存関係やソースコードを変更すると、同じローンチジョブでも別のバージョンが作成されます。
6. **Overrides** セクションで、あなたのローンチジョブに設定されている入力値を新しく指定できます。よく使われるオーバーライドには、エントリーポイントのコマンド、引数、または新しい W&B Run の `wandb.Run.config` で利用する値などがあります。  
  {{< img src="/images/launch/create_starter_queue_gs.png" alt="Queue configuration" >}}
  **Paste from...** ボタンをクリックすることで、同じローンチジョブを使った他の W&B Run から値をコピー＆ペーストできます。
7. **Queue** ドロップダウンから、このローンチジョブを追加したいローンチキュー名を選択します。
8. **Job Priority** ドロップダウンで、このローンチジョブの優先度を指定します。ローンチキューが優先度設定に対応していない場合、ジョブの優先度は「Medium」に設定されます。
9. **(オプション) チーム管理者によってキュー設定テンプレートが作成されている場合のみ**  
**Queue Configurations** フィールドには、チーム管理者が用意した設定オプションの値を入力します。  
たとえば、以下の例では、チーム管理者が利用できる AWS インスタンスタイプを設定しています。 チームメンバーは `ml.m4.xlarge` または `ml.p3.xlarge` のいずれかの計算インスタンスを選んでモデルの学習に利用できます。
{{< img src="/images/launch/team_member_use_config_template.png" alt="Config template selection" >}}
10. **Destination project** を選択します。ここに実行結果の Run が表示されます。このプロジェクトは、キューと同じエンティティに属している必要があります。
11. **Launch now** ボタンを選択します。

{{% /tab %}}
{{% tab "W&B CLI" %}}

`wandb launch` コマンドを使用してジョブをキューに追加します。ハイパーパラメーターの上書き用に JSON 設定ファイルを作成します。例えば [Quickstart]({{< relref path="../walkthrough.md" lang="ja" >}}) ガイドのスクリプトを使い、以下のようなオーバーライドを含む JSON ファイルを作成します。

```json title="config.json"
// このファイルはローンチ用の設定例です
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

キューの設定を上書きしたい場合や、あなたのローンチキューに設定リソースが定義されていない場合、config.json ファイルで `resource_args` キーを指定できます。上記の例を続けて、以下のように編集できます。

```json title="config.json"
// < > の中身はご自身の値に置き換えてください
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

`<>` で囲まれた部分は、あなた自身の値に置き換えてください。

`queue`（`-q`）フラグにはキュー名を、`job`（`-j`）フラグにはジョブ名を、`config`（`-c`）フラグには設定ファイルへのパスを指定してください。

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B Team をご利用の場合は、`entity` フラグ（`-e`）でキューに利用するエンティティを明示的に指定することをおすすめします。

{{% /tab %}}
{{% /tabpane %}}