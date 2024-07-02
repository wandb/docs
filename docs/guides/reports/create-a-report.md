---
description: W&B レポートは、アプリのUIを使用するか、または Weights & Biases SDK を使ってプログラムで作成します。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# レポートを作成する

<head>
  <title>Create a W&B Report</title>
</head>

W&B App UIを使用してインタラクティブに、またはW&B Python SDKを使用してプログラム的にレポートを作成します。

:::info
この[Google Colabの例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb)を参照してください。
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Report tab', value: 'reporttab'},
    {label: 'W&B Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

1. W&B Appのプロジェクトワークスペースに移動します。
2. ワークスペースの右上にある **Create report** をクリックします。

![Create report button](/images/reports/create_a_report_button.png)

3. モーダルが表示されます。開始したいチャートを選択します。チャートはレポートインターフェースから後で追加または削除できます。

![Create report modal](/images/reports/create_a_report_modal.png)

4. **Filter run sets** オプションを選択して、新しいRunがレポートに追加されないようにします。このオプションのオンオフを切り替えることができます。**Create report** をクリックすると、レポートタブにドラフトレポートが表示され、続けて作業できます。

  </TabItem>
  <TabItem value="reporttab">

1. W&B Appのプロジェクトワークスペースに移動します。
2. プロジェクト内の **Reports** タブを選択します。
3. レポートページで **Create Report** ボタンを選択します。

![Create report button](/images/reports/create_report_button.png)

  </TabItem>
  <TabItem value="sdk">


`wandb` ライブラリを使用してプログラム的にレポートを作成します。

```python
import wandb
import wandb_workspaces.reports.v2 as wr
```

Report Class Public API（[`wandb.apis.reports`](https://docs.wandb.ai/ref/python/public-api/api#reports)）を使用してレポートインスタンスを作成します。プロジェクトの名前を指定します。

```python
report = wr.Report(project="report_standard")
```

レポートは `.save()` メソッドを呼び出すまでW&Bサーバーにアップロードされません。

```python
report.save()
```

App UIを使用してインタラクティブに、またはプログラム的にレポートを編集する方法については [Edit a report](https://docs.wandb.ai/guides/reports/edit-a-report) を参照してください。
  </TabItem>
</Tabs>