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

W&B App UIを使ってインタラクティブに、またはW&B Python SDKを使ってプログラム的にレポートを作成できます。

:::info
この [Google Colabの例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) を参照してください。
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Report tab', value: 'reporttab'},
    {label: 'W&B Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

1. W&B Appでプロジェクトワークスペースに移動します。
2. ワークスペースの右上にある **Create report** をクリックします。

![](/images/reports/create_a_report_button.png)

3. モーダルが表示されます。開始するチャートを選択します。後でレポートインターフェースからチャートを追加または削除できます。

![](/images/reports/create_a_report_modal.png)

4. 新しい run がレポートに追加されないようにするには、**Filter run sets** オプションを選択します。このオプションはオンまたはオフに切り替えることができます。**Create report** をクリックすると、レポートタブにドラフトレポートが作成され、作業を続けることができます。

  </TabItem>
  <TabItem value="reporttab">

1. W&B Appでプロジェクトワークスペースに移動します。
2. プロジェクトの **Reports** タブ（クリップボードの画像）を選択します。
3. レポートページで **Create Report** ボタンを選択します。

![](/images/reports/create_report_button.png)
  </TabItem>
  <TabItem value="sdk">

`wandb` ライブラリを使ってプログラム的にレポートを作成します。

```python
import wandb
import wandb_workspaces.reports.v2 as wr
```

Report クラスのパブリック API（[`wandb.apis.reports`](https://docs.wandb.ai/ref/python/public-api/api#reports)）を使用してレポートインスタンスを作成します。プロジェクトの名前を指定してください。

```python
report = wr.Report(project="report_standard")
```

レポートは `.save()` メソッドを呼び出すまでW&Bサーバーにアップロードされません：

```python
report.save()
```

