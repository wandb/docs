---
description: アプリUIまたは Weights & Biases SDK を使用して W&B Report を作成します。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# レポートを作成する

<head>
  <title>Create a W&B Report</title>
</head>

レポートは、W&BアプリのUIを使って対話的に作成するか、W&B Python SDKを使用してプログラム的に作成できます。

:::info
Python SDKを使ってプログラム的にレポートを作成する機能は現在ベータ版であり、積極的に開発中です。例については、この[Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb)をご覧ください。
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Report tab', value: 'reporttab'},
    {label: 'Python SDK (Beta)', value: 'sdk'},
  ]}>
  <TabItem value="app">

1. W&Bアプリでプロジェクトワークスペースに移動します。
2. ワークスペースの右上隅にある**Create report**をクリックします。

![](/images/reports/create_a_report_button.png)

3. モーダルが表示されます。開始するチャートを選択します。後でレポートインターフェースからチャートを追加または削除できます。

![](/images/reports/create_a_report_modal.png)

4. **Filter run sets**オプションを選択して、新しいrunsがレポートに追加されないようにします。このオプションをオンまたはオフに切り替えることができます。**Create report**をクリックすると、ドラフトレポートがレポートタブに表示され、作業を続けることができます。

  </TabItem>
  <TabItem value="reporttab">

1. W&Bアプリでプロジェクトワークスペースに移動します。
2. プロジェクトの**Reports**タブ（クリップボードのアイコン）を選択します。
3. レポートページで**Create Report**ボタンを選択します。

![](/images/reports/create_report_button.png)
  </TabItem>
  <TabItem value="sdk">

`wandb`ライブラリを使用してプログラム的にレポートを作成します。

```python
import wandb
import wandb.apis.reports as wr
```

Public APIのReportクラス（[`wandb.apis.reports`](https://docs.wandb.ai/ref/python/public-api/api#reports)）を使用してレポートインスタンスを作成します。プロジェクトの名前を指定してください。

```python
report = wr.Report(project="report_standard")
```

レポートは、`.save()`メソッドを呼び出すまでW&Bサーバーにアップロードされません:

```python
report.save()
```

App UIを使用して対話的にレポートを編集する方法や、プログラム的にレポートを編集する方法についての情報は、[Edit a report](https://docs.wandb.ai/guides/reports/edit-a-report)をご覧ください。
  </TabItem>
</Tabs>