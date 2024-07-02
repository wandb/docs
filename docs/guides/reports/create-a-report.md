---
description: App UI または Weights & Biases SDK を使用して W&B Report を作成します。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# レポートを作成する

<head>
  <title>Create a W&B Report</title>
</head>

W&BアプリUIを使用して対話的に、またはW&B Python SDKを使用してプログラム的にレポートを作成します。

:::info
こちらの[Google Colabの例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb)をご覧ください。
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Report tab', value: 'reporttab'},
    {label: 'W&B Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

1. W&Bアプリでプロジェクトワークスペースに移動します。
2. ワークスペースの右上にある **Create report** をクリックします。

![](/images/reports/create_a_report_button.png)

3. モーダルが表示されます。最初に追加したいチャートを選択します。レポートインターフェースから後でチャートを追加または削除できます。

![](/images/reports/create_a_report_modal.png)

4. **Filter run sets** オプションを選択して、新しいrunがレポートに追加されないようにします。このオプションはオンまたはオフに切り替えることができます。**Create report** をクリックすると、レポートタブに下書きが表示され、作業を続けることができます。

  </TabItem>
  <TabItem value="reporttab">

1. W&Bアプリでプロジェクトワークスペースに移動します。
2. プロジェクトの**Reports**タブ（クリップボードの画像）を選択します。
3. レポートページの **Create Report** ボタンを選択します。

![](/images/reports/create_report_button.png)
  </TabItem>
  <TabItem value="sdk">

`wandb` ライブラリを使用してプログラム的にレポートを作成します。

```python
import wandb
import wandb_workspaces.reports.v2 as wr
```

ReportクラスのパブリックAPI（[`wandb.apis.reports`](https://docs.wandb.ai/ref/python/public-api/api#reports)）を使用してレポートインスタンスを作成します。プロジェクトの名前を指定します。

```python
report = wr.Report(project="report_standard")
```

レポートは、`.save()` メソッドを呼び出すまではW&Bサーバーにアップロードされません。

```python
report.save()
```

App UIを使用して対話的に、またはプログラム的にレポートを編集する方法については、[Edit a report](https://docs.wandb.ai/guides/reports/edit-a-report)をご覧ください。
  </TabItem>
</Tabs>