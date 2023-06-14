---
description: >-
  Create a W&B Report with the App UI or programmatically with the Weights &
  Biases SDK.
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# レポートを作成する

<head>
  <title>W&Bレポートを作成する</title>
</head>

`wandb` Python SDKを使ったプログラムによるレポート作成か、アプリUIでインタラクティブにレポートを作成します。

:::info
Python SDKを使ったレポートのプログラム作成は、ベータ版であり、積極的に開発が行われています。
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'アプリUI', value: 'app'},
    {label: 'レポートタブ', value: 'reporttab'},
    {label: 'Python SDK (ベータ)', value: 'sdk'},
  ]}>
  <TabItem value="app">

ワークスペースの右上隅にある**レポートを作成**をクリックします。

![](/images/reports/create_a_report_button.png)
はじめに選択したチャートを選んでください。後からレポートインターフェースからチャートを追加や削除できます。

![](/images/reports/create_a_report_modal.png)

**Filter run sets** オプションを選択して、新しいrunsがレポートに追加されるのを防ぎます。このオプションはオン・オフが切り替えられます。 **Create report** をクリックすると、レポートタブ内にドラフトレポートが用意され、作業を続けることができます。
  </TabItem>
  <TabItem value="reporttab">

プロジェクト内の **Reports** タブに移動し、レポートページ上の **Create Report** ボタンを選択してください。これで新しい空白のレポートが作成されます。レポートを保存して共有可能なリンクを取得するか、別のワークスペースや別のプロジェクトからチャートをレポートに送信します。

![](/images/reports/create_report_button.png)
  </TabItem>
  <TabItem value="sdk">

`wandb` ライブラリを利用してプログラムでレポートを作成します。

```python
import wandb
import wandb.apis.reports as wr

# レポート変更の誤操作を避けるためのW&B要件
wandb.require('report-editing')
```

レポートクラスの Public API（[`wandb.apis.reports`](https://docs.wandb.ai/ref/python/public-api/api#reports)）を使って、プロジェクトの名前を指定してレポートインスタンスを作成します。

```python
report = wr.Report(project='report_standard')
```
レポートは、.`save()`メソッドを呼び出すまで、Weights & Biasesサーバーにアップロードされません:



```python

report.save()

```



アプリUIまたはプログラムでレポートを編集する方法については、[レポートの編集](https://docs.wandb.ai/guides/reports/edit-a-report)を参照してください。

  </TabItem>

</Tabs>