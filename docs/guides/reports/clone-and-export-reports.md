---
title: Clone and export reports
description: W&B 리포트를 PDF 또는 LaTeX로 내보내기.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

## 리포트 내보내기

리포트를 PDF 또는 LaTeX로 내보냅니다. 리포트 내에서 케밥 아이콘을 선택하여 드롭다운 메뉴를 확장합니다. **다운로드**를 선택하고 PDF 또는 LaTeX 출력 형식을 선택합니다.

## 리포트 복제

<Tabs
  defaultValue="app"
  values={[
    {label: '앱 UI', value: 'app'},
    {label: 'Python SDK', value: 'python'}
  ]}>
  <TabItem value="app">

리포트 내에서 케밥 아이콘을 선택하여 드롭다운 메뉴를 확장합니다. **이 리포트 복제** 버튼을 선택합니다. 복제할 리포트의 대상을 모달에서 선택합니다. **리포트 복제**를 선택합니다.

![](/images/reports/clone_reports.gif)

리포트를 복제하여 프로젝트의 템플릿과 형식을 재사용합니다. 팀 계정 내에서 프로젝트를 복제하면 복제된 프로젝트는 팀이 볼 수 있습니다. 개인 계정 내에서 프로젝트를 복제하면 해당 사용자만 볼 수 있습니다.
  </TabItem>
  <TabItem value="python">

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb"></CTAButtons>

URL에서 Report를 불러와 템플릿으로 사용합니다.

```python
report = wr.Report(
    project=PROJECT, title="Quickstart Report", description="That was easy!"
)  # 생성
report.save()  # 저장
new_report = wr.Report.from_url(report.url)  # 불러오기
```

`new_report.blocks`의 내용을 편집합니다.

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
    report.blocks[:1] + [wr.H1("패널 그리드 예제"), pg] + report.blocks[1:]
)
new_report.save()
```
  </TabItem>
</Tabs>