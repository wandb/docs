---
description: W&BレポートをPDFまたはLaTeX形式でエクスポートする。
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# レポートのクローンとエクスポート

<head>
  <title>W&Bレポートのクローンとエクスポート</title>
</head>

## レポートのエクスポート

レポートをPDFまたはLaTeX形式でエクスポートします。レポート内でケバブアイコンを選択してドロップダウンメニューを展開し、**ダウンロード**を選択して、PDFまたはLaTeXの出力形式を選択します。

## レポートのクローニング

<Tabs
  defaultValue="app"
  values={[
    {label: 'アプリUI', value: 'app'},
    {label: 'Python SDK', value: 'python'}
  ]}>
  <TabItem value="app">

レポート内でケバブアイコンを選択してドロップダウンメニューを展開し、**このレポートを複製**ボタンを選択します。モーダルで複製したレポートの保存先を選択し、**レポートを複製**を選択します。

![](@site/static/images/reports/clone_reports.gif)

プロジェクトのテンプレートとフォーマットを再利用するためにレポートを複製します。チームのアカウント内でプロジェクトを複製した場合、複製したプロジェクトはチームのメンバーに表示されます。個人のアカウント内で複製されたプロジェクトは、そのユーザーにしか表示されません。
  </TabItem>
  <TabItem value="python">
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/report\_api)

URLからレポートをロードして、テンプレートとして使用します。

```python
report = wr.Report(
    project=PROJECT,
    title='クイックスタート レポート',
    description="簡単だった！"
)                                              # 作成
report.save()                                  # 保存
new_report = wr.Report.from_url(report.url)    # ロード
```

`new_report.blocks`の中身を編集します。

```python
pg = wr.PanelGrid(
    runsets=[
        wr.Runset(ENTITY, PROJECT, "最初のRunセット"),
        wr.Runset(ENTITY, PROJECT, "象のみ！", query="elephant"),
    ],
    panels=[
        wr.LinePlot(x='Step', y=['val_acc'], smoothing_factor=0.8),
        wr.BarPlot(metrics=['acc']),
        wr.MediaBrowser(media_keys='img', num_columns=1),
        wr.RunComparer(diff_only='split', layout={'w': 24, 'h': 9}),
    ]
)
new_report.blocks = report.blocks[:1] + [wr.H1("パネルグリッドの例"), pg] + report.blocks[1:]
new_report.save()
```
  </TabItem>
</Tabs>