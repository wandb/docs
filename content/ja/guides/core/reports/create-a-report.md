---
title: Reports を作成
description: W&B App を使用するか、プログラムから W&B Reports を作成します。
menu:
  default:
    identifier: ja-guides-core-reports-create-a-report
    parent: reports
weight: 10
---

{{% alert %}}
W&B Report and Workspace API は公開プレビュー中です。
{{% /alert %}}

以下のタブを選択して、W&B App で Reports を作成する方法、または W&B Report and Workspace API を使用してプログラムで作成する方法を学んでください。

プログラムによる Reports 作成の例は、こちらの [Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) を参照してください。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. W&B App で Projects Workspace に移動します。
2. Workspace の右上隅にある **Create report** をクリックします。

   {{< img src="/images/reports/create_a_report_button.png" alt="Create report ボタン" >}}

3. モーダルが表示されます。開始するチャートを選択してください。後で report インターフェースからチャートを追加または削除できます。

    {{< img src="/images/reports/create_a_report_modal.png" alt="Create report モーダル" >}}

4. **Filter run sets** オプションを選択して、新しい Runs が report に追加されないようにします。このオプションはオン/オフを切り替えられます。**Create report** をクリックすると、ドラフト report が report タブで利用可能になり、作業を続けられます。
{{% /tab %}}

{{% tab header="Report タブ" value="reporttab"%}}
1. W&B App で Projects Workspace に移動します。
2. 自分の Projects で **Reports** タブ (クリップボードのアイコン) を選択します。
3. report ページで **Create Report** ボタンを選択します。

   {{< img src="/images/reports/create_report_button.png" alt="Create report ボタン" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
プログラムで report を作成します:

1. W&B SDK (`wandb`) と Report and Workspace API (`wandb-workspaces`) をインストールします:
    ```bash
    pip install wandb wandb-workspaces
    ```
2. 次に、モジュールをインポートします:
    ```python
    import wandb
    import wandb_workspaces.reports.v2 as wr
    ```
3. `wandb_workspaces.reports.v2.Report` を使用して report を作成します。Report クラスのパブリック API ([`wandb.apis.reports`]({{< relref path="/ref/python/public-api/api.md#reports" lang="ja" >}})) を使って report インスタンスを作成します。Projects の名前を指定します。
    ```python
    report = wr.Report(project="report_standard")
    ```

4. report を保存します。`.save()` メソッドを呼び出すまで、Reports は W&B サーバーにアップロードされません:
    ```python
    report.save()
    ```

App UI またはプログラムで report をインタラクティブに編集する方法については、[report を編集する]({{< relref path="/guides/core/reports/edit-a-report" lang="ja" >}}) を参照してください。
{{% /tab %}}
{{< /tabpane >}}