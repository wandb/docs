---
title: レポートを作成する
description: W&B アプリまたはプログラムから W&B Report を作成する方法をご紹介します。
menu:
  default:
    identifier: create-a-report
    parent: reports
weight: 10
---

{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

以下のタブから、W&B App でレポートを作成する方法、または W&B Report および Workspace API を使ってプログラムでレポートを作成する方法を確認できます。

プログラムからレポートを作成する例については、この [Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) をご覧ください。

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. W&B App の自身のプロジェクト ワークスペースに移動します。
2. ワークスペースの右上にある **Create report** をクリックします。

   {{< img src="/images/reports/create_a_report_button.png" alt="Create report button" >}}

3. モーダルが表示されます。最初に含めたいチャートを選択してください。これらのチャートは後からレポート画面で追加や削除が可能です。

    {{< img src="/images/reports/create_a_report_modal.png" alt="Create report modal" >}}

4. **Filter run sets** オプションを選択すると、新しい Run がレポートに自動的に追加されるのを防げます。このオプションはいつでもオン・オフを切り替えられます。**Create report** をクリックすると、下書きのレポートがレポートタブに保存され、後で作業を続けられます。
{{% /tab %}}

{{% tab header="Report tab" value="reporttab"%}}
1. W&B App の自身のプロジェクト ワークスペースに移動します。
2. プロジェクト内の **Reports** タブ（クリップボードのアイコン）を選択します。
3. レポートページ上の **Create Report** ボタンをクリックします。

   {{< img src="/images/reports/create_report_button.png" alt="Create report button" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
プログラムからレポートを作成する手順です：

1. W&B SDK（`wandb`）および Report and Workspace API（`wandb-workspaces`）をインストールします。
    ```bash
    pip install wandb wandb-workspaces
    ```
2. 次に、workspaces をインポートします
    ```python
    import wandb
    import wandb_workspaces.reports.v2 as wr
    ```
3. `wandb_workspaces.reports.v2.Report` を使ってレポートを作成します。Report クラスパブリックAPI（[`wandb.apis.reports`]({{< relref "/ref/python/public-api/api.md#reports" >}})）を利用してインスタンスを作成し、プロジェクト名を指定します。
    ```python
    report = wr.Report(project="report_standard")
    ```
4. レポートを保存します。. `save()` メソッドを呼ぶまで、レポートは W&B サーバーにアップロードされません。
    ```python
    report.save()
    ```

App UI を使って対話的に、またはプログラムからレポートを編集する方法については [Edit a report]({{< relref "/guides/core/reports/edit-a-report" >}}) をご参照ください。
{{% /tab %}}
{{< /tabpane >}}