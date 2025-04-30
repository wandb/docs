---
title: レポートを作成する
description: W&B レポートは、App UI を使用するか Weights & Biases SDK を使ってプログラムで作成します。
menu:
  default:
    identifier: ja-guides-core-reports-create-a-report
    parent: reports
weight: 10
---

レポートを作成するには、W&B App UI をインタラクティブに使用するか、W&B Python SDK を使用してプログラムで行います。

{{% alert %}}
この[Google Colab の例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb)を参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}
1. W&B App でプロジェクトワークスペースに移動します。
2. ワークスペースの右上隅にある **Create report** をクリックします。

   {{< img src="/images/reports/create_a_report_button.png" alt="" >}}

3. モーダルが表示されます。最初に使用したいチャートを選択します。レポートのインターフェースから後でチャートを追加または削除することができます。

    {{< img src="/images/reports/create_a_report_modal.png" alt="" >}}

4. **Filter run sets** オプションを選択すると、新しい run がレポートに追加されるのを防げます。このオプションをオンまたはオフに切り替えることができます。**Create report** をクリックすると、レポートタブにドラフトレポートが表示され、作業を続けることができます。
{{% /tab %}}

{{% tab header="Report tab" value="reporttab"%}}
1. W&B App でプロジェクトワークスペースに移動します。
2. プロジェクト内の **Reports** タブ（クリップボードの画像）を選択します。
3. レポートページで **Create Report** ボタンを選択します。 

   {{< img src="/images/reports/create_report_button.png" alt="" >}}
{{% /tab %}}

{{% tab header="W&B Python SDK" value="sdk"%}}
`wandb` ライブラリを使用してプログラムでレポートを作成します。

1. W&B SDK と Workspaces API をインストールします:
    ```bash
    pip install wandb wandb-workspaces
    ```
2. 次に、ワークスペースをインポートします。
    ```python
    import wandb
    import wandb_workspaces.reports.v2 as wr
    ```       
3. `wandb_workspaces.reports.v2.Report` を使用してレポートを作成します。Report Class Public API ([`wandb.apis.reports`]({{< relref path="/ref/python/public-api/api#reports" lang="ja" >}})) を使ってレポートのインスタンスを作成します。プロジェクトに名前を指定します。  
    ```python
    report = wr.Report(project="report_standard")
    ```  

4. レポートを保存します。レポートは、`save()` メソッドを呼び出すまで W&B サーバーにアップロードされません。
    ```python
    report.save()
    ```

App UI を使用してインタラクティブに、またはプログラムでレポートを編集する方法については、[Edit a report]({{< relref path="/guides/core/reports/edit-a-report" lang="ja" >}}) を参照してください。
{{% /tab %}}
{{< /tabpane >}}