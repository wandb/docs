---
title: Clone and export reports
description: W&B レポートを PDF または LaTeX としてエクスポートする。
menu:
  default:
    identifier: ja-guides-core-reports-clone-and-export-reports
    parent: reports
weight: 40
---

## レポートのエクスポート

レポートを PDF または LaTeX としてエクスポートします。レポート内でケバブアイコンを選択してドロップダウンメニューを展開します。**Download and** を選び、PDF もしくは LaTeX 出力形式を選択します。

## レポートのクローン

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}
レポート内でケバブアイコンを選択してドロップダウンメニューを展開します。**Clone this report** ボタンを選びます。モーダル内でクローンするレポートの行き先を選びます。**Clone report** を選択します。

{{< img src="/images/reports/clone_reports.gif" alt="" >}}

_Project_ のテンプレートとフォーマットを再利用するためにレポートをクローンします。_Project_ がチームのアカウント内でクローンされた場合、そのチームはクローンされた _Project_ を閲覧できます。個人のアカウント内でクローンされたプロジェクトはそのユーザーにのみ見えます。
{{% /tab %}}

{{% tab header="Python SDK" value="python"%}}



URL からレポートを読み込み、テンプレートとして使用します。

```python
report = wr.Report(
    project=PROJECT, title="Quickstart Report", description="That was easy!"
)  # 作成
report.save()  # 保存
new_report = wr.Report.from_url(report.url)  # 読み込み
```

`new_report.blocks` 内の内容を編集します。

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
{{% /tab %}}
{{< /tabpane >}}