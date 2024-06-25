---
description: W&BレポートをPDFまたはLaTeXとしてエクスポートします。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Clone and export reports

<head>
  <title>Clone and export W&B Reports</title>
</head>

## レポートのエクスポート

レポートをPDFまたはLaTeX形式でエクスポートします。レポート内でケバブアイコンを選択し、ドロップダウンメニューを展開します。**Download** を選択し、PDFまたはLaTeXの出力形式を選びます。

## レポートのクローン

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'python'}
  ]}>
  <TabItem value="app">

レポート内でケバブアイコンを選択し、ドロップダウンメニューを展開します。**Clone this report** ボタンを選択します。モーダルでクローンするレポートの保存先を選びます。**Clone report** を選択します。

![](@site/static/images/reports/clone_reports.gif)

プロジェクトのテンプレートとフォーマットを再利用するためにレポートをクローンします。チームのアカウント内でプロジェクトをクローンすると、クローンされたプロジェクトはチームメンバーに表示されます。個人のアカウント内でクローンされたプロジェクトは、そのユーザーのみに表示されます。
  </TabItem>
  <TabItem value="python">

[**Try in a Colab Notebook here →**](http://wandb.me/report\_api)

URLからレポートを読み込み、テンプレートとして使用します。

```python
report = wr.Report(
    project=PROJECT, title="Quickstart Report", description="That was easy!"
)  # 作成
report.save()  # 保存
new_report = wr.Report.from_url(report.url)  # 読み込み
```

`new_report.blocks`内の内容を編集します。

```python
pg = wr.PanelGrid(
    runsets=[
        wr.Runset(ENTITY, PROJECT, "First Run Set"),
        wr.Runset(ENTITY, PROJECT, "Elephants Only!", query="elephant"),
    ],
    panels=[
        wr.LinePlot(x="Step", y=["val_acc"], smoothing_factor=0.8),
        wr.BarPlot(metrics=["acc"]),
        wr.MediaBrowser(media_keys="img", num_columns=1),
        wr.RunComparer(diff_only="split", layout={"w": 24, "h": 9}),
    ],
)
new_report.blocks = (
    report.blocks[:1] + [wr.H1("Panel Grid Example"), pg] + report.blocks[1:]
)
new_report.save()
```
  </TabItem>
</Tabs>