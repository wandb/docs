---
title: Edit a report
description: App UI로 리포트를 대화식으로 편집하거나 W&B SDK를 사용하여 프로그래밍 방식으로 편집하세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

리포트를 App UI로 대화형으로 편집하거나 W&B SDK를 사용하여 프로그래밍 방식으로 편집합니다.

Reports는 _blocks_로 구성됩니다. Blocks는 리포트의 본문을 구성합니다. 이 블록 내에서 텍스트, 이미지, 임베디드 시각화, Experiments 및 run의 플롯, 그리고 패널 그리드를 추가할 수 있습니다.

_패널 그리드_ 는 패널과 _run 세트_를 보유하는 특정 유형의 블록입니다. Run 세트는 W&B의 프로젝트에 로그된 runs의 컬렉션입니다. 패널은 run 세트 데이터의 시각화입니다.

:::tip
[Programmatic workspaces tutorial](../../tutorials/workspaces.md)을 확인하여 저장된 워크스페이스 보기를 만들고 사용자 정의하는 방법에 대한 단계별 예제를 확인하세요.
:::

:::info
프로그램 방식으로 리포트를 편집하려면 W&B Python SDK 외에도 `wandb-workspaces`를 설치해야 합니다:

pip install wandb wandb-workspaces
:::

### 플롯 추가

각 패널 그리드에는 run 세트와 패널 세트가 있습니다. 섹션 하단의 run 세트는 그리드 내 패널에 표시될 데이터를 제어합니다. 다른 run 세트에서 데이터를 끌어오는 차트를 추가하려면 새 패널 그리드를 만드세요.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Workspaces API', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. **Add panel**을 선택하여 패널을 추가하세요. W&B에서 지원하는 모든 패널을 추가할 수 있으며, 이것에는 라인 플롯, 산포도 또는 병렬 좌표 차트가 포함됩니다.

![Add charts to a report](/images/reports/demo_report_add_panel_grid.gif)
  
  </TabItem>
  <TabItem value="sdk">

SDK를 사용하여 프로그래밍 방식으로 리포트에 플롯을 추가합니다. `PanelGrid` Public API Class에서 `panels` 파라미터에 하나 이상의 플롯 또는 차트 오브젝트의 리스트를 전달하세요. 관련 Python Class로 플롯 또는 차트 오브젝트를 생성하세요.

다음 예제는 라인 플롯과 산포도를 생성하는 방법을 시연합니다.

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

프로그램 방식으로 리포트에 추가할 수 있는 사용 가능한 플롯 및 차트에 대한 자세한 정보는 `wr.panels`를 참조하세요.
  </TabItem>
</Tabs>

### Run 세트 추가

프로젝트에서 run 세트를 App UI 또는 W&B SDK로 대화형으로 추가합니다.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Workspaces API', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. 드롭다운에서 패널 그리드를 선택하세요. 이렇게 하면 리포트가 생성된 프로젝트로부터 run 세트가 자동으로 가져와집니다.
  </TabItem>
  <TabItem value="sdk">

`wr.Runset()` 및 `wr.PanelGrid` Classes를 사용하여 프로젝트에서 run 세트를 추가하세요. 다음 프로세스는 runset을 추가하는 방법을 설명합니다:

1. `wr.Runset()` 오브젝트 인스턴스를 만드세요. 프로젝트 파라미터에 runsets를 포함한 프로젝트 이름을 제공하고, 엔티티 파라미터에는 프로젝트 소유자를 제공하세요.
2. `wr.PanelGrid()` 오브젝트 인스턴스를 만드세요. 하나 이상의 runset 오브젝트를 `runsets` 파라미터에 전달하세요.
3. 하나 이상의 `wr.PanelGrid()` 오브젝트 인스턴스를 리스트에 저장하세요.
4. 패널 그리드 인스턴스의 리스트로 보고서 인스턴스 블록 속성 업데이트합니다.

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

하나의 호출로 SDK에 runsets와 패널을 추가할 수 있습니다:

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
  </TabItem>
</Tabs>

### 코드 블록 추가

App UI 또는 W&B SDK를 사용하여 리포트에 코드 블록을 대화형으로 추가합니다.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Workspaces API', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. 드롭다운에서 **Code**를 선택하세요.

코드 블록의 오른쪽에서 프로그래밍 언어의 이름을 선택하세요. 그러면 드롭다운이 확장됩니다. 드롭다운에서 프로그래밍 언어 구문을 선택하세요. JavaScript, Python, CSS, JSON, HTML, Markdown 및 YAML 중에서 선택할 수 있습니다.
  </TabItem>
  <TabItem value="sdk">

`wr.CodeBlock` Class를 사용하여 프로그래밍 방식으로 코드 블록을 생성하세요. 각각의 language 및 code 파라미터에 표시할 언어의 이름과 코드를 제공하세요.

예를 들어, 다음 예제는 YAML 파일의 리스트를 시연합니다:

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

이는 다음과 유사한 코드 블록을 렌더링합니다:

```yaml
this:
- is
- a
cool:
- yaml
- file
```

다음 예제는 Python 코드 블록을 시연합니다:

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

이는 다음과 유사한 코드 블록을 렌더링합니다:

```md
Hello, World!
```
  </TabItem>
</Tabs>

### Markdown

App UI 또는 W&B SDK를 사용하여 리포트에 markdown을 대화형으로 추가합니다.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Workspaces API', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. 드롭다운에서 **Markdown**을 선택하세요.
  </TabItem>
  <TabItem value="sdk">

`wandb.apis.reports.MarkdownBlock` Class를 사용하여 프로그램 방식으로 markdown 블록을 생성하세요. `text` 파라미터에 문자열을 전달하세요:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

이는 다음과 유사한 markdown 블록을 렌더링합니다:

![](/images/reports/markdown.png)
  </TabItem>
</Tabs>

### HTML 요소

App UI 또는 W&B SDK를 사용하여 리포트에 HTML 요소를 대화형으로 추가합니다.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Workspaces API', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에 슬래시(`/`)를 입력하여 드롭다운 메뉴를 표시합니다. 드롭다운에서 텍스트 블록 유형을 선택합니다. 예를 들어, H2 헤딩 블록을 생성하려면 `Heading 2` 옵션을 선택합니다.
  </TabItem>
  <TabItem value="sdk">

하나 이상의 HTML 요소 리스트를 `wandb.apis.reports.blocks` 속성에 전달합니다. 다음 예제는 H1, H2 및 순서 없는 리스트를 생성하는 방법을 시연합니다:

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

이는 다음과 같은 HTML 요소로 렌더링될 것입니다:


![](/images/reports/render_html.png)

  </TabItem>
</Tabs>

### 리치 미디어 링크 임베드

리포트 내에서 App UI 또는 W&B SDK를 사용하여 리치 미디어를 임베드합니다.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Workspaces API', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에 URL을 복사하여 붙여 넣어 리치 미디어를 임베드하세요. 다음 애니메이션은 Twitter, YouTube 및 SoundCloud에서 URL을 복사하고 붙여넣는 방법을 시연합니다.

#### Twitter

리포트에 Tweet 링크 URL을 복사하여 붙여 넣으면 리포트에서 Tweet을 볼 수 있습니다.

![](/images/reports/twitter.gif)

####

#### Youtube

리포트에 YouTube 동영상 URL 링크를 복사하여 붙여 넣어 동영상을 임베드합니다.

![](/images/reports/youtube.gif)

#### SoundCloud

리포트에 SoundCloud 링크를 복사하여 붙여 넣어 오디오 파일을 임베드합니다.

![](/images/reports/soundcloud.gif)
  </TabItem>
  <TabItem value="sdk">

하나 이상의 임베드된 미디어 오브젝트 리스트를 `wandb.apis.reports.blocks` 속성에 전달합니다. 다음 예제는 동영상 및 Twitter 미디어를 리포트에 임베드하는 방법을 시연합니다:

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
  </TabItem>
</Tabs>

### 패널 그리드 복사 및 삭제

다시 사용하고 싶은 레이아웃이 있으면, 패널 그리드를 선택하여 동일한 리포트 또는 다른 리포트에 복사-붙여 넣기 할 수 있습니다.

우측 상단 모서리의 드래그 핸들을 선택하여 전체 패널 그리드 섹션을 강조 표시하세요. 패널 그리드, 텍스트 및 헤딩과 같은 리포트 내 영역을 강조 표시하고 선택하려면 클릭 드래그하세요.

![](/images/reports/demo_copy_and_paste_a_panel_grid_section.gif)

패널 그리드를 선택하고, 키보드에서 `delete`를 눌러 패널 그리드를 삭제하세요.

![](/images/reports/delete_panel_grid.gif)

### Reports 구성하기 위한 헤더 접기

리포트에서 헤더를 접어 내용이 텍스트 블록 내에 숨겨지도록 합니다. 리포트가 로드되면 확장된 헤더만 내용이 표시됩니다. 헤더를 보고서에서 접으면 내용을 구성하고 과도한 데이터 로드를 방지할 수 있습니다. 다음 gif는 이 과정을 보여줍니다.

![](/images/reports/collapse_headers.gif)