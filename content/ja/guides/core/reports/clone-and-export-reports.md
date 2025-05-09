---
title: レポートをクローンしてエクスポートする
description: W&B レポートを PDF または LaTeX としてエクスポートします。
menu:
  default:
    identifier: ja-guides-core-reports-clone-and-export-reports
    parent: reports
weight: 40
---

## レポートをエクスポート

レポートを PDF または LaTeX としてエクスポートします。レポート内でケバブアイコンを選択し、ドロップダウンメニューを展開します。**Download** を選択し、PDF または LaTeX の出力形式を選択します。

## レポートをクローン

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}
レポート内でケバブアイコンを選択し、ドロップダウンメニューを展開します。**Clone this report** ボタンを選択します。モーダルでクローンされたレポートの保存先を選択します。**Clone report** を選択します。

{{< img src="/images/reports/clone_reports.gif" alt="" >}}

プロジェクトのテンプレートと形式を再利用するためにレポートをクローンします。チームのアカウント内でプロジェクトをクローンした場合、クローンされたプロジェクトはチームに表示されます。個人のアカウント内でプロジェクトをクローンした場合、そのプロジェクトはそのユーザーのみに表示されます。
{{% /tab %}}

{{% tab header="Python SDK" value="python"%}}




テンプレートとして使用するために URL からレポートをロードします。

```python
report = wr.Report(
    project=PROJECT, title="Quickstart Report", description="That was easy!"
)  # 作成
report.save()  # 保存
new_report = wr.Report.from_url(report.url)  # ロード
```

`new_report.blocks` 内のコンテンツを編集します。

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