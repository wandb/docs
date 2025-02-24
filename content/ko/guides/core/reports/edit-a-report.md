---
title: Edit a report
description: App UI를 사용하여 또는 W&B SDK를 사용하여 프로그래밍 방식으로 리포트 를 대화형으로 편집합니다.
menu:
  default:
    identifier: ko-guides-core-reports-edit-a-report
    parent: reports
weight: 20
---

App UI 또는 W&B SDK를 통해 프로그래밍 방식으로 리포트를 대화형으로 편집합니다.

Reports는 _블록_으로 구성됩니다. 블록은 리포트의 본문을 구성합니다. 이러한 블록 내에서 텍스트, 이미지, 내장된 시각화, 실험 및 run의 플롯, 패널 그리드를 추가할 수 있습니다.

_패널 그리드_는 패널과 _run sets_을 담는 특정 유형의 블록입니다. Run sets는 W&B의 프로젝트에 기록된 runs 모음입니다. 패널은 run set 데이터의 시각화입니다.

{{% alert %}}
저장된 워크스페이스 뷰를 생성하고 사용자 정의하는 방법에 대한 단계별 예제는 [프로그래밍 방식 워크스페이스 튜토리얼]({{< relref path="/tutorials/workspaces.md" lang="ko" >}})을 확인하세요.
{{% /alert %}}

{{% alert %}}
리포트를 프로그래밍 방식으로 편집하려면 W&B Python SDK 외에 `wandb-workspaces`가 설치되어 있는지 확인하세요.

```pip
pip install wandb wandb-workspaces
```
{{% /alert %}}

## 플롯 추가

각 패널 그리드에는 run sets 집합과 패널 집합이 있습니다. 섹션 하단의 run sets는 그리드의 패널에 표시되는 데이터를 제어합니다. 다른 runs 집합에서 데이터를 가져오는 차트를 추가하려면 새 패널 그리드를 만드세요.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. **패널 추가**를 선택하여 패널을 추가합니다. 선 플롯, 산점도 또는 평행 좌표 차트를 포함하여 W&B에서 지원하는 모든 패널을 추가할 수 있습니다.

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="리포트에 차트 추가" >}}
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk"%}}
SDK를 사용하여 프로그래밍 방식으로 리포트에 플롯을 추가합니다. `PanelGrid` Public API Class의 `panels` 파라미터에 하나 이상의 플롯 또는 차트 오브젝트 목록을 전달합니다. 관련 Python Class를 사용하여 플롯 또는 차트 오브젝트를 만듭니다.

다음 예제에서는 선 플롯과 산점도를 만드는 방법을 보여줍니다.

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

blocks = [
    wr.PanelGrid(
        panels=[
            wr.LinePlot(x="time", y="velocity"),
            wr.ScatterPlot(x="time", y="acceleration"),
        ]
    )
]

report.blocks = blocks
report.save()
```

프로그래밍 방식으로 리포트에 추가할 수 있는 사용 가능한 플롯 및 차트에 대한 자세한 내용은 `wr.panels`를 참조하세요.

{{% /tab %}}
{{< /tabpane >}}

## Run sets 추가

App UI 또는 W&B SDK를 사용하여 프로젝트에서 run sets를 대화형으로 추가합니다.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. 드롭다운에서 패널 그리드를 선택합니다. 그러면 리포트가 생성된 프로젝트에서 run set가 자동으로 임포트됩니다.

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk"%}}

`wr.Runset()` 및 `wr.PanelGrid` Class를 사용하여 프로젝트에서 run sets를 추가합니다. 다음 절차에서는 runset를 추가하는 방법을 설명합니다.

1. `wr.Runset()` 오브젝트 인스턴스를 만듭니다. 프로젝트 파라미터에 대한 runsets를 포함하는 프로젝트 이름과 엔티티 파라미터에 대한 프로젝트를 소유한 엔티티 이름을 제공합니다.
2. `wr.PanelGrid()` 오브젝트 인스턴스를 만듭니다. 하나 이상의 runset 오브젝트 목록을 `runsets` 파라미터에 전달합니다.
3. 하나 이상의 `wr.PanelGrid()` 오브젝트 인스턴스를 목록에 저장합니다.
4. 패널 그리드 인스턴스 목록으로 리포트 인스턴스 블록 속성을 업데이트합니다.

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

panel_grids = wr.PanelGrid(
    runsets=[wr.RunSet(project="<project-name>", entity="<entity-name>")]
)

report.blocks = [panel_grids]
report.save()
```

선택적으로 SDK에 대한 하나의 호출로 runsets와 패널을 추가할 수 있습니다.

```python
import wandb

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

panel_grids = wr.PanelGrid(
    panels=[
        wr.LinePlot(
            title="line title",
            x="x",
            y=["y"],
            range_x=[0, 100],
            range_y=[0, 100],
            log_x=True,
            log_y=True,
            title_x="x axis title",
            title_y="y axis title",
            ignore_outliers=True,
            groupby="hyperparam1",
            groupby_aggfunc="mean",
            groupby_rangefunc="minmax",
            smoothing_factor=0.5,
            smoothing_type="gaussian",
            smoothing_show_original=True,
            max_runs_to_show=10,
            plot_type="stacked-area",
            font_size="large",
            legend_position="west",
        ),
        wr.ScatterPlot(
            title="scatter title",
            x="y",
            y="y",
            # z='x',
            range_x=[0, 0.0005],
            range_y=[0, 0.0005],
            # range_z=[0,1],
            log_x=False,
            log_y=False,
            # log_z=True,
            running_ymin=True,
            running_ymean=True,
            running_ymax=True,
            font_size="small",
            regression=True,
        ),
    ],
    runsets=[wr.RunSet(project="<project-name>", entity="<entity-name>")],
)


report.blocks = [panel_grids]
report.save()
```

{{% /tab %}}
{{< /tabpane >}}

## Run set 고정

리포트는 프로젝트에서 최신 데이터를 표시하도록 run sets를 자동으로 업데이트합니다. 해당 run set를 *고정*하여 리포트에서 run set를 보존할 수 있습니다. run set를 고정하면 특정 시점에 리포트에서 run set의 상태를 보존합니다.

리포트를 볼 때 run set를 고정하려면 **필터** 버튼 근처의 패널 그리드에서 눈송이 아이콘을 클릭하세요.

{{< img src="/images/reports/freeze_runset.png" alt="" >}}

## 코드 블록 추가

App UI 또는 W&B SDK를 사용하여 리포트에 코드 블록을 대화형으로 추가합니다.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. 드롭다운에서 **코드**를 선택합니다.

코드 블록 오른쪽에 있는 프로그래밍 언어 이름을 선택합니다. 그러면 드롭다운이 확장됩니다. 드롭다운에서 프로그래밍 언어 구문을 선택합니다. Javascript, Python, CSS, JSON, HTML, Markdown 및 YAML 중에서 선택할 수 있습니다.

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

`wr.CodeBlock` Class를 사용하여 프로그래밍 방식으로 코드 블록을 만듭니다. 각각 language 및 code 파라미터에 대해 표시할 언어 이름과 코드를 제공합니다.

예를 들어, 다음 예제에서는 YAML 파일의 목록을 보여줍니다.

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.CodeBlock(
        code=["this:", "- is", "- a", "cool:", "- yaml", "- file"], language="yaml"
    )
]

report.save()
```

그러면 다음과 유사한 코드 블록이 렌더링됩니다.

```yaml
this:
- is
- a
cool:
- yaml
- file
```

다음 예제에서는 Python 코드 블록을 보여줍니다.

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

그러면 다음과 유사한 코드 블록이 렌더링됩니다.

```md
Hello, World!
```

{{% /tab %}}

{{% /tabpane %}}

## Markdown 추가

App UI 또는 W&B SDK를 사용하여 리포트에 Markdown을 대화형으로 추가합니다.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. 드롭다운에서 **Markdown**을 선택합니다.

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

`wandb.apis.reports.MarkdownBlock` Class를 사용하여 프로그래밍 방식으로 Markdown 블록을 만듭니다. 문자열을 `text` 파라미터에 전달합니다.

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

그러면 다음과 유사한 Markdown 블록이 렌더링됩니다.

{{< img src="/images/reports/markdown.png" alt="" >}}

{{% /tab %}}

{{% /tabpane %}}

## HTML 요소 추가

App UI 또는 W&B SDK를 사용하여 리포트에 HTML 요소를 대화형으로 추가합니다.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. 드롭다운에서 텍스트 블록 유형을 선택합니다. 예를 들어 H2 제목 블록을 만들려면 `Heading 2` 옵션을 선택합니다.

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

하나 이상의 HTML 요소 목록을 `wandb.apis.reports.blocks` 속성에 전달합니다. 다음 예제에서는 H1, H2 및 순서가 지정되지 않은 목록을 만드는 방법을 보여줍니다.

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.H1(text="How Programmatic Reports work"),
    wr.H2(text="Heading 2"),
    wr.UnorderedList(items=["Bullet 1", "Bullet 2"]),
]

report.save()
```

그러면 HTML 요소가 다음으로 렌더링됩니다.

{{< img src="/images/reports/render_html.png" alt="" >}}

{{% /tab %}}

{{% /tabpane %}}

## 리치 미디어 링크 임베드

App UI 또는 W&B SDK를 사용하여 리포트 내에 리치 미디어를 임베드합니다.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

URL을 복사하여 리포트에 붙여넣어 리포트 내에 리치 미디어를 임베드합니다. 다음 애니메이션에서는 Twitter, YouTube 및 SoundCloud에서 URL을 복사하여 붙여넣는 방법을 보여줍니다.

### Twitter

트윗 링크 URL을 복사하여 리포트에 붙여넣어 리포트 내에서 트윗을 봅니다.

{{< img src="/images/reports/twitter.gif" alt="" >}}

### Youtube

YouTube 비디오 URL 링크를 복사하여 붙여넣어 리포트에 비디오를 임베드합니다.

{{< img src="/images/reports/youtube.gif" alt="" >}}

### SoundCloud

SoundCloud 링크를 복사하여 붙여넣어 오디오 파일을 리포트에 임베드합니다.

{{< img src="/images/reports/soundcloud.gif" alt="" >}}

{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

하나 이상의 임베디드 미디어 오브젝트 목록을 `wandb.apis.reports.blocks` 속성에 전달합니다. 다음 예제에서는 비디오 및 Twitter 미디어를 리포트에 임베드하는 방법을 보여줍니다.

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
    wr.Twitter(
        embed_html='<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The voice of an angel, truly. <a href="https://twitter.com/hashtag/MassEffect?src=hash&amp;ref_src=twsrc%5Etfw">#MassEffect</a> <a href="https://t.co/nMev97Uw7F">pic.twitter.com/nMev97Uw7F</a></p>&mdash; Mass Effect (@masseffect) <a href="https://twitter.com/masseffect/status/1428748886655569924?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>\n'
    ),
]
report.save()
```

{{% /tab %}}

{{% /tabpane %}}

## 패널 그리드 복제 및 삭제

재사용하려는 레이아웃이 있는 경우 패널 그리드를 선택하고 복사하여 붙여넣어 동일한 리포트에서 복제하거나 다른 리포트에 붙여넣을 수도 있습니다.

오른쪽 상단 모서리에 있는 드래그 핸들을 선택하여 전체 패널 그리드 섹션을 강조 표시합니다. 클릭하고 드래그하여 패널 그리드, 텍스트 및 제목과 같은 리포트에서 영역을 강조 표시하고 선택합니다.

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="" >}}

패널 그리드를 선택하고 키보드에서 `delete`를 눌러 패널 그리드를 삭제합니다.

{{< img src="/images/reports/delete_panel_grid.gif" alt="" >}}

## 제목을 축소하여 Reports 구성

리포트에서 제목을 축소하여 텍스트 블록 내의 콘텐츠를 숨깁니다. 리포트가 로드되면 확장된 제목만 콘텐츠를 표시합니다. 리포트에서 제목을 축소하면 콘텐츠를 구성하고 과도한 데이터 로드를 방지할 수 있습니다. 다음 gif에서는 프로세스를 보여줍니다.

{{< img src="/images/reports/collapse_headers.gif" alt="" >}}
