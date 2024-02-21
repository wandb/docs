---
description: Export a W&B Report as a PDF or LaTeX.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 리포트 복제 및 내보내기

<head>
  <title>W&B 리포트 복제 및 내보내기</title>
</head>

## 리포트 내보내기

리포트를 PDF 또는 LaTeX로 내보냅니다. 리포트에서 케밥 아이콘을 선택하여 드롭다운 메뉴를 확장합니다. **다운로드**를 선택하고 PDF 또는 LaTeX 출력 형식을 선택하세요.

## 리포트 복제하기

<Tabs
  defaultValue="app"
  values={[
    {label: '앱 UI', value: 'app'},
    {label: '파이썬 SDK', value: 'python'}
  ]}>
  <TabItem value="app">

리포트에서 케밥 아이콘을 선택하여 드롭다운 메뉴를 확장합니다. **이 리포트 복제하기** 버튼을 선택합니다. 모달에서 복제된 리포트의 목적지를 선택합니다. **리포트 복제**를 선택하세요.

![](@site/static/images/reports/clone_reports.gif)

프로젝트의 템플릿과 형식을 재사용하기 위해 리포트를 복제하세요. 팀 계정 내에서 프로젝트를 복제하면 복제된 프로젝트가 팀에게 보입니다. 개인 계정 내에서 복제된 프로젝트는 해당 사용자에게만 보입니다.
  </TabItem>
  <TabItem value="python">

[**여기에서 Colab 노트북에서 시도해보세요 →**](http://wandb.me/report\_api)

템플릿으로 사용할 URL에서 리포트를 로드합니다.

```python
report = wr.Report(
    project=PROJECT, title="퀵스타트 리포트", description="쉽네요!"
)  # 생성
report.save()  # 저장
new_report = wr.Report.from_url(report.url)  # 로드
```

`new_report.blocks` 내의 내용을 편집합니다.

```python
pg = wr.PanelGrid(
    runsets=[
        wr.Runset(ENTITY, PROJECT, "첫 실행 세트"),
        wr.Runset(ENTITY, PROJECT, "코끼리만!", query="elephant"),
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