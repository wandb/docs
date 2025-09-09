---
title: Reports をクローンしてエクスポート
description: W&B Reports を PDF または LaTeX としてエクスポートします。
menu:
  default:
    identifier: ja-guides-core-reports-clone-and-export-reports
    parent: reports
weight: 40
---

{{% alert %}}
W&B Report and Workspace API はパブリックプレビュー版です。
{{% /alert %}}

## Reports をエクスポートする

Report を PDF または LaTeX としてエクスポートします。Report 内で、ケバブアイコンを選択してドロップダウンメニューを展開します。**ダウンロード** を選択し、PDF または LaTeX のいずれかの出力形式を選択します。

## Reports のクローン作成

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
Report 内で、ケバブアイコンを選択してドロップダウンメニューを展開します。**この Report をクローン** ボタンを選択します。モーダルで、クローンされた Report の保存先を選択します。**Report をクローン** を選択します。

{{< img src="/images/reports/clone_reports.gif" alt="Reports のクローン作成" >}}

Report をクローンして、Projects のテンプレートとフォーマットを再利用します。チームのアカウント内で Projects をクローンした場合、クローンされた Projects はチームに表示されます。個人のアカウント内でクローンされた Projects は、そのユーザーのみに表示されます。
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

URL から Report を読み込み、テンプレートとして使用します。

```python
report = wr.Report(
    project=PROJECT, title="Quickstart Report", description="That was easy!"
)  # 作成
report.save()  # 保存
new_report = wr.Report.from_url(report.url)  # 読み込み
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