---
description: W&B レポートは、アプリのUIを使用して作成するか、Weights & Biases SDKを使ってプログラムで作成することができます。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# レポートを作成する

<head>
  <title>W&B レポートを作成</title>
</head>

レポートは W&B アプリの UI を使ってインタラクティブに作成するか、W&B Python SDK を使ってプログラム的に作成できます。

:::info
こちらの [Google Colab の例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) を参照してください。
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Report tab', value: 'reporttab'},
    {label: 'W&B Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

1. W&B アプリでプロジェクトワークスペースに移動します。
2. ワークスペースの右上にある **Create report** をクリックします。

![](/images/reports/create_a_report_button.png)

3. モーダルが表示されます。開始するチャートを選択します。チャートの追加や削除は後でレポートインターフェースから行えます。

![](/images/reports/create_a_report_modal.png)

4. **Filter run sets** オプションを選択して、新しい run がレポートに追加されないようにします。このオプションはオンまたはオフに切り替え可能です。**Create report** をクリックすると、レポートタブに下書きレポートが作成され、作業を続けることができます。


  </TabItem>
  <TabItem value="reporttab">

1. W&B アプリでプロジェクトワークスペースに移動します。
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

Report クラスの Public API を使用してレポートインスタンスを作成します ([`wandb.apis.reports`](https://docs.wandb.ai/ref/python/public-api/api#reports))。プロジェクトの名前を指定します。

```python
report = wr.Report(project="report_standard")
```

レポートは .`save()` メソッドを呼び出すまで W&B サーバーにアップロードされません：

```python
report.save()
```

アプリの UI を使ってインタラクティブにレポートを編集する方法やプログラム的に編集する方法については、[レポートを編集する](https://docs.wandb.ai/guides/reports/edit-a-report) を参照してください。
  </TabItem>
</Tabs>
