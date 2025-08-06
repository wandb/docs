---
title: Reports 복제 및 내보내기
description: W&B 리포트를 PDF 또는 LaTeX 형식으로 내보낼 수 있습니다.
menu:
  default:
    identifier: ko-guides-core-reports-clone-and-export-reports
    parent: reports
weight: 40
---

{{% alert %}}
W&B Report 및 Workspace API는 Public Preview 단계입니다.
{{% /alert %}}

## 리포트 내보내기

리포트를 PDF 또는 LaTeX 형식으로 내보낼 수 있습니다. 리포트 내에서 케밥 아이콘(세 점 아이콘)을 선택해 드롭다운 메뉴를 엽니다. **Download and**를 선택한 후 PDF 또는 LaTeX 출력 형식을 고르세요.

## 리포트 복제하기

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
리포트 내에서 케밥 아이콘을 선택해 드롭다운 메뉴를 펼칩니다. **Clone this report** 버튼을 선택하세요. 모달에서 복제할 리포트의 목적지를 지정한 뒤, **Clone report**를 선택하세요.

{{< img src="/images/reports/clone_reports.gif" alt="Cloning reports" >}}

리포트를 복제하면 해당 프로젝트의 템플릿과 형식을 재사용할 수 있습니다. 팀 계정 내에서 프로젝트를 복제할 경우, 복제된 프로젝트는 팀 전체에서 볼 수 있습니다. 개인 계정 내에서 복제하면 해당 사용자만 볼 수 있습니다.
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

URL에서 Report를 불러와 템플릿으로 사용할 수 있습니다.

```python
report = wr.Report(
    project=PROJECT, title="Quickstart Report", description="That was easy!"
)  # 생성
report.save()  # 저장
new_report = wr.Report.from_url(report.url)  # 불러오기
```

`new_report.blocks` 내에서 내용을 수정할 수 있습니다.

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