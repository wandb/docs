---
title: Clone and export reports
description: W&B 리포트 를 PDF 또는 LaTeX로 내보내세요.
menu:
  default:
    identifier: ko-guides-core-reports-clone-and-export-reports
    parent: reports
weight: 40
---

## Reports 내보내기

리포트를 PDF 또는 LaTeX로 내보냅니다. 리포트 내에서 케밥 아이콘을 선택하여 드롭다운 메뉴를 확장합니다. **다운로드**를 선택하고 PDF 또는 LaTeX 출력 형식을 선택합니다.

## Reports 복제

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}
리포트 내에서 케밥 아이콘을 선택하여 드롭다운 메뉴를 확장합니다. **이 리포트 복제** 버튼을 선택합니다. 모달에서 복제된 리포트의 대상을 선택합니다. **리포트 복제**를 선택합니다.

{{< img src="/images/reports/clone_reports.gif" alt="" >}}

프로젝트의 템플릿과 형식을 재사용하기 위해 리포트를 복제합니다. 팀 계정 내에서 프로젝트를 복제하면 복제된 프로젝트가 팀에 표시됩니다. 개인 계정 내에서 복제된 프로젝트는 해당 사용자에게만 표시됩니다.
{{% /tab %}}

{{% tab header="Python SDK" value="python"%}}

URL에서 Report를 로드하여 템플릿으로 사용합니다.

```python
report = wr.Report(
    project=PROJECT, title="Quickstart Report", description="That was easy!"
)  # Create
report.save()  # Save
new_report = wr.Report.from_url(report.url)  # Load
```

`new_report.blocks` 내에서 콘텐츠를 편집합니다.

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
