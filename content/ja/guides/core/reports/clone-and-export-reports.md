---
title: レポートをクローンおよびエクスポートする
description: W&B レポートを PDF または LaTeX 形式でエクスポートする。
menu:
  default:
    identifier: ja-guides-core-reports-clone-and-export-reports
    parent: reports
weight: 40
---

{{% alert %}}
W&B Report および Workspace API はパブリックプレビュー中です。
{{% /alert %}}

## レポートのエクスポート

レポートを PDF または LaTeX 形式でエクスポートできます。レポート内でケバブアイコンを選択し、ドロップダウンメニューを開きます。**ダウンロード** を選び、PDF または LaTeX の出力形式を選択してください。

## レポートのクローン

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
レポート内でケバブアイコンを選択し、ドロップダウンメニューを展開します。**このレポートをクローン** ボタンを選択します。表示されるモーダルでクローン先を選択してください。**レポートをクローン** を押します。

{{< img src="/images/reports/clone_reports.gif" alt="Cloning reports" >}}

レポートをクローンすると、プロジェクトのテンプレートやフォーマットを再利用できます。チームアカウント内でプロジェクトをクローンすると、チームのメンバー全員がそのプロジェクトを閲覧できます。個人アカウント内でクローンした場合は、そのユーザーのみに表示されます。
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

Report の URL からテンプレートとして読み込むことができます。

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
    report.blocks[:1] + [wr.H1("パネルグリッド例"), pg] + report.blocks[1:]
)
new_report.save()
```
{{% /tab %}}
{{< /tabpane >}}