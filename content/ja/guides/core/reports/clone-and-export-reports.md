---
title: レポートのクローンとエクスポート
description: W&B レポートを PDF または LaTeX としてエクスポートします。
menu:
  default:
    identifier: clone-and-export-reports
    parent: reports
weight: 40
---

{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

## レポートのエクスポート

レポートを PDF または LaTeX 形式でエクスポートできます。レポート画面内でケバブアイコンをクリックし、ドロップダウンメニューを展開します。**ダウンロード** を選択し、PDF か LaTeX の出力形式を選んでください。

## レポートのクローン作成

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
レポート画面内でケバブアイコンをクリックし、ドロップダウンメニューを展開します。**このレポートをクローン** ボタンを選択します。モーダルでクローン先を指定し、**レポートをクローン** を選択してください。

{{< img src="/images/reports/clone_reports.gif" alt="Cloning reports" >}}

レポートをクローンすることで、プロジェクトのテンプレートやフォーマットを再利用できます。チームアカウント内でプロジェクトをクローンすると、そのクローンされたプロジェクトはチーム全体から閲覧可能です。個人アカウント内でクローンした場合は、そのユーザーのみに表示されます。
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

Report の URL から読み込んで、テンプレートとして利用できます。

```python
report = wr.Report(
    project=PROJECT, title="Quickstart Report", description="That was easy!"
)  # 作成
report.save()  # 保存
new_report = wr.Report.from_url(report.url)  # ロード
```

`new_report.blocks` で内容を編集できます。

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