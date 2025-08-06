---
title: レポートを作成する
description: W&B アプリまたはプログラムによって W&B Report を作成します。
menu:
  default:
    identifier: ja-guides-core-reports-create-a-report
    parent: reports
weight: 10
---

{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

以下のタブを選択して、W&B App でレポートを作成する方法、または W&B Report と Workspace API でプログラムからレポートを作成する方法を学んでください。

プログラムからレポートを作成する例については、[Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) をご覧ください。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. W&B App で自分のプロジェクト Workspace に移動します。
2. Workspace の右上にある **Create report** をクリックします。

   {{< img src="/images/reports/create_a_report_button.png" alt="Create report button" >}}

3. モーダルウィンドウが表示されます。最初に追加したいチャートを選択してください。チャートは後からレポート画面で追加・削除が可能です。

    {{< img src="/images/reports/create_a_report_modal.png" alt="Create report modal" >}}

4. **Filter run sets** オプションを選択することで、新しい run がレポートに追加されるのを防ぐことができます。このオプションはオン/オフの切り替えが可能です。**Create report** をクリックすると、レポートタブに下書きのレポートが作成され、作業を続けられます。
{{% /tab %}}

{{% tab header="Report tab" value="reporttab"%}}
1. W&B App で自分のプロジェクト Workspace に移動します。
2. プロジェクト内の **Reports** タブ（クリップボードのアイコン）を選択します。
3. レポートページで **Create Report** ボタンをクリックします。

   {{< img src="/images/reports/create_report_button.png" alt="Create report button" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
プログラムからレポートを作成する方法：

1. W&B SDK（`wandb`）と Report and Workspace API（`wandb-workspaces`）をインストールします。
    ```bash
    pip install wandb wandb-workspaces
    ```
2. 次に workspaces をインポートします。
    ```python
    import wandb
    import wandb_workspaces.reports.v2 as wr
    ```       
3. `wandb_workspaces.reports.v2.Report` でレポートを作成します。Report クラスのパブリック API（[`wandb.apis.reports`]({{< relref path="/ref/python/public-api/api.md#reports" lang="ja" >}})）でインスタンスを生成します。プロジェクト名も指定してください。   
    ```python
    report = wr.Report(project="report_standard")
    ```  

4. レポートを保存します。レポートは `.save()` メソッドを呼ぶまで W&B サーバーにアップロードされません。
    ```python
    report.save()
    ```

App UI でレポートをインタラクティブに編集する方法や、プログラムから編集する方法については [Edit a report]({{< relref path="/guides/core/reports/edit-a-report" lang="ja" >}}) をご覧ください。
{{% /tab %}}
{{< /tabpane >}}