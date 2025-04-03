---
title: Add job to queue
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-add-job-to-queue
    parent: create-and-deploy-jobs
url: guides/launch/add-job-to-queue
---

次のページでは、Launch キューに Launch ジョブを追加する方法について説明します。

{{% alert %}}
あなたまたはあなたのチームの誰かがすでに Launch キューを設定していることを確認してください。詳細については、[Launch の設定]({{< relref path="/launch/set-up-launch/" lang="ja" >}})のページを参照してください。
{{% /alert %}}

## キューにジョブを追加する

W&B App を使用してインタラクティブに、または W&B CLI を使用してプログラムで、ジョブをキューに追加します。

{{< tabpane text=true >}}
{{% tab "W&B app" %}}
W&B App を使用して、プログラムでジョブをキューに追加します。

1. W&B の Project ページに移動します。
2. 左側のパネルで、**Jobs** アイコンを選択します。
  {{< img src="/images/launch/project_jobs_tab_gs.png" alt="" >}}
3. **Jobs** ページには、以前に実行された W&B の run から作成された W&B の Launch ジョブのリストが表示されます。
  {{< img src="/images/launch/view_jobs.png" alt="" >}}
4. ジョブ名の横にある **Launch** ボタンを選択します。モーダルがページの右側に表示されます。
5. **Job version** ドロップダウンから、使用する Launch ジョブのバージョンを選択します。Launch ジョブは、他の [W&B Artifact]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}}) と同様にバージョン管理されています。ジョブの実行に使用されるソフトウェアの依存関係またはソースコードを変更すると、同じ Launch ジョブの異なるバージョンが作成されます。
6. **Overrides** セクション内で、Launch ジョブに設定されているすべての入力に新しい値を指定します。一般的なオーバーライドには、新しいエントリポイントコマンド、引数、または新しい W&B の run の `wandb.config` の値が含まれます。
  {{< img src="/images/launch/create_starter_queue_gs.png" alt="" >}}
  **Paste from...** ボタンをクリックして、Launch ジョブを使用した他の W&B の run から値をコピーして貼り付けることができます。
7. **Queue** ドロップダウンから、Launch ジョブを追加する Launch キューの名前を選択します。
8. **Job Priority** ドロップダウンを使用して、Launch ジョブの優先度を指定します。Launch キューが優先順位付けをサポートしていない場合、Launch ジョブの優先度は「Medium」に設定されます。
9. **（オプション）チーム管理者がキュー構成テンプレートを作成した場合にのみ、この手順に従ってください**
**Queue Configurations** フィールド内で、チームの管理者によって作成された構成オプションの値を指定します。
たとえば、次の例では、チーム管理者はチームが使用できる AWS インスタンスタイプを構成しました。この場合、チームメンバーは `ml.m4.xlarge` または `ml.p3.xlarge` コンピュートインスタンスタイプのいずれかを選択して、モデルをトレーニングできます。
{{< img src="/images/launch/team_member_use_config_template.png" alt="" >}}
10. 結果の run が表示される **Destination project** を選択します。この Project は、キューと同じエンティティに属している必要があります。
11. **Launch now** ボタンを選択します。

{{% /tab %}}
{{% tab "W&B CLI" %}}

`wandb launch` コマンドを使用して、ジョブをキューに追加します。ハイパー パラメーターのオーバーライドを含む JSON 構成を作成します。たとえば、[クイックスタート]({{< relref path="../walkthrough.md" lang="ja" >}}) ガイドのスクリプトを使用して、次のオーバーライドを含む JSON ファイルを作成します。

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
JSON 構成ファイルを提供しない場合、W&B Launch はデフォルトの パラメータ を使用します。
{{% /alert %}}

キュー構成をオーバーライドする場合、または Launch キューに構成リソースが定義されていない場合は、config.json ファイルで `resource_args` キーを指定できます。たとえば、上記の例に続いて、config.json ファイルは次のようになります。

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

`<>` 内の値を独自の値に置き換えます。

`queue`(`-q`) フラグのキューの名前、`job`(`-j`) フラグのジョブの名前、および `config`(`-c`) フラグの構成ファイルへのパスを指定します。

```bash
wandb launch -j <job> -q <queue-name> \
-e <entity-name> -c path/to/config.json
```
W&B の Teams で作業する場合は、キューが使用するエンティティを示すために `entity` フラグ (`-e`) を指定することをお勧めします。

{{% /tab %}}
{{% /tabpane %}}
