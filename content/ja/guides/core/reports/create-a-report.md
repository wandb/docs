---
title: Create a report
description: W&B の App UI を使用するか、Weights & Biases SDK を使用してプログラムで W&B Report を作成します。
menu:
  default:
    identifier: ja-guides-core-reports-create-a-report
    parent: reports
weight: 10
---

W&B App UI または W&B Python SDK を使用して、プログラムでインタラクティブに report を作成します。

{{% alert %}}
例については、こちらの [Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) をご覧ください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}
1. W&B App で project の workspace に移動します。
2. workspace の右上隅にある [**Create report**] をクリックします。

   {{< img src="/images/reports/create_a_report_button.png" alt="" >}}

3. モーダルが表示されます。最初に表示するグラフを選択します。グラフは report インターフェイスから後で追加または削除できます。

    {{< img src="/images/reports/create_a_report_modal.png" alt="" >}}

4. [**Filter run sets**] オプションを選択して、新しい run が report に追加されないようにします。このオプションはオン/オフを切り替えることができます。[**Create report**] をクリックすると、下書き report が report タブで使用できるようになり、作業を続けることができます。
{{% /tab %}}

{{% tab header="Report tab" value="reporttab"%}}
1. W&B App で project の workspace に移動します。
2. project の [**Reports**] タブ (クリップボードの画像) を選択します。
3. report ページで [**Create Report**] ボタンを選択します。

   {{< img src="/images/reports/create_report_button.png" alt="" >}}
{{% /tab %}}

{{% tab header="W&B Python SDK" value="sdk"%}}
`wandb` ライブラリを使用して、プログラムで report を作成します。

1. W&B SDK と Workspaces API をインストールします。
    ```bash
    pip install wandb wandb-workspaces
    ```
2. 次に、ワークスペースをインポートします。
    ```python
    import wandb
    import wandb_workspaces.reports.v2 as wr
    ```       
3. `wandb_workspaces.reports.v2.Report` を使用して report を作成します。Report Class Public API ([`wandb.apis.reports`]({{< relref path="/ref/python/public-api/api#reports" lang="ja" >}})) を使用して report インスタンスを作成します。 project の名前を指定します。
    ```python
    report = wr.Report(project="report_standard")
    ```

4. report を保存します。Reports は、.`save()` メソッドを呼び出すまで W&B サーバーにアップロードされません。
    ```python
    report.save()
    ```

App UI またはプログラムで report をインタラクティブに編集する方法については、[Edit a report]({{< relref path="/guides/core/reports/edit-a-report" lang="ja" >}}) を参照してください。
{{% /tab %}}
{{< /tabpane >}}
