---
title: Create a report
description: W&B レポートを作成するには、アプリ UI を使用するか、Weights & Biases SDK を使用してプログラムで作成します。
menu:
  default:
    identifier: ja-guides-core-reports-create-a-report
    parent: reports
weight: 10
---

レポートを対話的に W&B App UI で作成するか、プログラムで W&B Python SDK を使用して作成します。

{{% alert %}}
この [Google Colab の例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) をご覧ください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}
1. W&B App であなたのプロジェクト ワークスペースに移動します。
2. ワークスペースの右上隅にある **Create report** をクリックします。

   {{< img src="/images/reports/create_a_report_button.png" alt="" >}}

3. モーダルが表示されます。開始したいチャートを選択してください。後でレポート インターフェイスからチャートを追加または削除できます。

    {{< img src="/images/reports/create_a_report_modal.png" alt="" >}}

4. **Filter run sets** オプションを選択して、新しい run がレポートに追加されないようにします。このオプションはオンまたはオフに切り替えることができます。**Create report** をクリックすると、編集を続けるためのドラフト レポートがレポート タブに表示されます。
{{% /tab %}}

{{% tab header="Report tab" value="reporttab"%}}
1. W&B App であなたのプロジェクト ワークスペースに移動します。
2. プロジェクトの **Reports** タブ (クリップボード画像) を選択します。
3. レポート ページで **Create Report** ボタンを選択します。

   {{< img src="/images/reports/create_report_button.png" alt="" >}}
{{% /tab %}}

{{% tab header="W&B Python SDK" value="sdk"%}}
`wandb` ライブラリを使用してプログラムでレポートを作成します。

1. W&B SDK と Workspaces API をインストールします:
    ```bash
    pip install wandb wandb-workspaces
    ```
2. 次に、workspaces をインポートします
    ```python
    import wandb
    import wandb_workspaces.reports.v2 as wr
    ```       
3. `wandb_workspaces.reports.v2.Report` を使用してレポートを作成します。Report クラス Public API ([`wandb.apis.reports`]({{< relref path="/ref/python/public-api/api#reports" lang="ja" >}})) を使用してレポート インスタンスを作成します。プロジェクトの名前を指定してください。
    ```python
    report = wr.Report(project="report_standard")
    ```

4. レポートを保存します。Reports は `.save()` メソッドを呼び出すまで W&B サーバーにアップロードされません:
    ```python
    report.save()
    ```

App UI を使用して対話的にまたはプログラムでレポートを編集する方法については、[レポートの編集]({{< relref path="/guides/core/reports/edit-a-report" lang="ja" >}}) を参照してください。
{{% /tab %}}
{{< /tabpane >}}