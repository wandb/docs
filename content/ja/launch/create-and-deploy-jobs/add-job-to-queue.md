---
title: Add job to queue
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-add-job-to-queue
    parent: create-and-deploy-jobs
url: guides/launch/add-job-to-queue
---

次のページでは、 Launch ジョブを Launch キューに追加する方法について説明します。

{{% alert %}}
あなたまたはあなたのチームの誰かが、すでに Launch キューを構成していることを確認してください。詳細については、[Launch のセットアップ]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

## ジョブをキューに追加する

W&B App を使用してインタラクティブに、または W&B CLI を使用してプログラムで、ジョブをキューに追加します。

{{< tabpane text=true >}}
{{% tab "W&B app" %}}
W&B App を使用して、プログラムでジョブをキューに追加します。

1. W&B Project ページに移動します。
2. 左側の パネル で [**Jobs**] アイコンを選択します。
  {{< img src="/images/launch/project_jobs_tab_gs.png" alt="" >}}
3. [**Jobs**] ページには、以前に実行された W&B run から作成された W&B Launch ジョブ のリストが表示されます。
  {{< img src="/images/launch/view_jobs.png" alt="" >}}
4. ジョブ名の横にある [**Launch**] ボタンを選択します。モーダルがページの右側に表示されます。
5. [**Job バージョン**] ドロップダウンから、使用する Launch ジョブ の バージョン を選択します。 Launch ジョブ は、他の [W&B Artifact]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}}) と同様に バージョン管理 されています。同じ Launch ジョブ の異なる バージョン は、ジョブ の実行に使用されるソフトウェアの依存関係またはソース コードを変更した場合に作成されます。
6. [**Overrides**] セクション内で、 Launch ジョブ に構成されている任意の入力の新しい 値 を指定します。一般的なオーバーライドには、新しいエントリポイント コマンド、 引数 、または新しい W&B run の `wandb.config` の 値 が含まれます。
  {{< img src="/images/launch/create_starter_queue_gs.png" alt="" >}}
  [**Paste from...**] ボタンをクリックして、 Launch ジョブ を使用した他の W&B run から 値 をコピーして貼り付けることができます。
7. [**Queue**] ドロップダウンから、 Launch ジョブ を追加する Launch キュー の名前を選択します。
8. [**Job Priority**] ドロップダウンを使用して、 Launch ジョブ の優先度を指定します。 Launch ジョブ の優先度は、 Launch キュー が優先順位付けをサポートしていない場合、「Medium」に設定されます。
9. **（オプション）チーム管理者がキュー構成テンプレートを作成した場合にのみ、この手順に従ってください**
[**Queue Configurations**] フィールド内で、チームの管理者によって作成された構成オプションの 値 を指定します。
たとえば、次の例では、チーム管理者はチームが使用できる AWS インスタンス タイプを構成しました。この場合、チームメンバーは `ml.m4.xlarge` または `ml.p3.xlarge` コンピューティング インスタンス タイプを選択して、モデル をトレーニングできます。
{{< img src="/images/launch/team_member_use_config_template.png" alt="" >}}
10. 結果の run が表示される [**Destination project**] を選択します。この プロジェクト は、キュー と同じエンティティに属している必要があります。
11. [**Launch now**] ボタンを選択します。

{{% /tab %}}
{{% tab "W&B CLI" %}}

`wandb launch` コマンド を使用して、ジョブ を キュー に追加します。 ハイパーパラメーター のオーバーライドを含む JSON 構成を作成します。たとえば、[クイックスタート]({{< relref path="../walkthrough.md" lang="ja" >}}) ガイド の スクリプト を使用して、次のオーバーライドを含む JSON ファイルを作成します。

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
JSON 構成ファイル を指定しない場合、W&B Launch はデフォルトの パラメータ を使用します。
{{% /alert %}}

キュー 構成をオーバーライドする場合、または Launch キュー に構成リソースが定義されていない場合は、config.json ファイル で `resource_args` キー を指定できます。たとえば、上記の例に続いて、config.json ファイル は次のようになります。

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

`<>` 内の 値 を独自の 値 に置き換えます。

`queue`(`-q`) フラグ に キュー の名前、`job`(`-j`) フラグ に ジョブ の名前、`config`(`-c`) フラグ に 構成ファイル への パス を指定します。

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B Team で作業する場合は、`entity` フラグ (`-e`) を指定して、キュー が使用するエンティティを示すことをお勧めします。

{{% /tab %}}
{{% /tabpane %}}
