---
description: Edit a report interactively with the App UI or programmatically with
  the W&B SDK.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 리포트 편집하기

<head>
  <title>Edit a W&B Report</title>
</head>


App UI 또는 W&B SDK를 사용하여 리포트를 대화식으로 또는 프로그래밍 방식으로 편집하세요.

리포트는 _블록_으로 구성됩니다. 블록은 리포트의 본문을 구성합니다. 이 블록 안에서 텍스트, 이미지, 내장 시각화, 실험 및 run으로부터의 플롯, 그리고 패널 그리드를 추가할 수 있습니다.

_패널 그리드_는 패널과 _run sets_를 보유하는 특정 유형의 블록입니다. Run sets는 W&B 프로젝트에 기록된 run의 모음입니다. 패널은 run set 데이터의 시각화입니다.

:::info
Python SDK를 사용한 리포트 프로그래밍 방식 편집은 베타 버전이며 활발히 개발 중입니다.
:::

### 플롯 추가하기

각 패널 그리드는 run sets와 패널 세트를 가지고 있습니다. 섹션 하단의 run sets는 그리드의 패널에 어떤 데이터가 표시될지를 제어합니다. 다른 run 집합에서 데이터를 가져오는 차트를 추가하려면 새 패널 그리드를 만드세요.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에서 슬래시(`/`)를 입력하면 드롭다운 메뉴가 표시됩니다. **패널 추가**를 선택하여 패널을 추가하세요. W&B에서 지원하는 모든 패널을 추가할 수 있습니다; 라인 플롯, 산점도 또는 병렬 좌표 차트를 포함합니다.



![리포트에 차트 추가하기](/images/reports/demo_report_add_panel_grid.gif)
  
  </TabItem>
  <TabItem value="sdk">

SDK를 사용하여 리포트에 프로그래밍 방식으로 플롯을 추가하세요. 하나 이상의 플롯 또는 차트 오브젝트 목록을 `PanelGrid` 공개 API 클래스의 `panels` 파라미터에 전달하세요. 각각의 Python 클래스로 플롯 또는 차트 오브젝트를 생성하세요.



다음 예제는 라인 플롯과 산점도를 생성하는 방법을 보여줍니다.

```python
import wandb
import wandb.apis.reports as wr

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

프로그래밍 방식으로 리포트에 추가할 수 있는 사용 가능한 플롯과 차트에 대한 자세한 정보는 `wr.panels`를 참조하세요.
  </TabItem>
</Tabs>

### run sets 추가하기

App UI나 W&B SDK를 사용하여 프로젝트에서 run sets를 대화식으로 추가하세요.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에서 슬래시(`/`)를 입력하면 드롭다운 메뉴가 표시됩니다. 드롭다운에서 Panel Grid를 선택하세요. 이렇게 하면 리포트가 생성된 프로젝트에서 run set이 자동으로 가져와집니다.
  </TabItem>
  <TabItem value="sdk">

`wr.Runset()` 및 `wr.PanelGrid` 클래스를 사용하여 프로젝트에서 run sets를 추가하세요. 다음 절차는 runset을 추가하는 방법을 설명합니다:

1. `wr.Runset()` 오브젝트 인스턴스를 생성하세요. 프로젝트 파라미터에는 runsets가 포함된 프로젝트의 이름을 제공하고, 엔티티 파라미터에는 프로젝트를 소유한 엔티티의 이름을 제공하세요.
2. `wr.PanelGrid()` 오브젝트 인스턴스를 생성하세요. `runsets` 파라미터에 하나 이상의 runset 오브젝트 목록을 전달하세요.
3. 하나 이상의 `wr.PanelGrid()` 오브젝트 인스턴스를 목록에 저장하세요.
4. 리포트 인스턴스의 blocks 속성을 패널 그리드 인스턴스 목록으로 업데이트하세요.

```python
import wandb
import wandb.apis.reports as wr

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

원하는 경우 SDK에 대한 한 번의 호출로 runsets와 패널을 추가할 수 있습니다:

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

### 코드 블록 추가하기

App UI 또는 W&B SDK를 사용하여 리포트에 코드 블록을 대화식으로 추가하세요.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에서 슬래시(`/`)를 입력하면 드롭다운 메뉴가 표시됩니다. 드롭다운에서 **코드**를 선택하세요.

코드 블록의 오른쪽에 프로그래밍 언어의 이름을 선택하세요. 이렇게 하면 드롭다운이 표시됩니다. 드롭다운에서 프로그래밍 언어 구문을 선택하세요. Javascript, Python, CSS, JSON, HTML, Markdown, YAML에서 선택할 수 있습니다.
  </TabItem>
  <TabItem value="sdk">

`wr.CodeBlock` 클래스를 사용하여 프로그래밍 방식으로 코드 블록을 생성하세요. 언어와 코드 파라미터에 각각 언어의 이름과 표시하려는 코드를 제공하세요.

예를 들어 다음 예제는 YAML 파일의 목록을 보여줍니다:

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.CodeBlock(
        code=["this:", "- is", "- a", "cool:", "- yaml", "- file"], language="yaml"
    )
]

report.save()
```

이것은 다음과 유사한 코드 블록을 렌더링합니다:

```yaml
this:
- is
- a
cool:
- yaml
- file
```

다음 예제는 Python 코드 블록을 보여줍니다:

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

이것은 다음과 유사한 코드 블록을 렌더링합니다:

```md
Hello, World!
```
  </TabItem>
</Tabs>

### 마크다운

App UI 또는 W&B SDK를 사용하여 리포트에 마크다운을 대화식으로 추가하세요.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에서 슬래시(`/`)를 입력하면 드롭다운 메뉴가 표시됩니다. 드롭다운에서 **Markdown**을 선택하세요.
  </TabItem>
  <TabItem value="sdk">

`wandb.apis.reports.MarkdownBlock` 클래스를 사용하여 프로그래밍 방식으로 마크다운 블록을 생성하세요. `text` 파라미터에 문자열을 전달하세요:

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

이것은 다음과 유사한 마크다운 블록을 렌더링합니다:

![](/images/reports/markdown.png)
  </TabItem>
</Tabs>

### HTML 요소

App UI 또는 W&B SDK를 사용하여 리포트에 HTML 요소를 추가하세요.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트에서 슬래시(`/`)를 입력하면 드롭다운 메뉴가 표시됩니다. 드롭다운에서 텍스트 블록 유형을 선택하세요. 예를 들어, H2 제목 블록을 만들려면 `제목 2` 옵션을 선택하세요.
  </TabItem>
  <TabItem value="sdk">

하나 이상의 HTML 요소를 `wandb.apis.reports.blocks` 속성에 목록으로 전달하세요. 다음 예제는 H1, H2 및 순서 없는 목록을 생성하는 방법을 보여줍니다:

```python
import wandb
import wandb.apis.reports as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.H1(text="How Programmatic Reports work"),
    wr.H2(text="Heading 2"),
    wr.UnorderedList(items=["Bullet 1", "Bullet 2"]),
]

report.save()
```

이것은 다음과 같은 HTML 요소를 렌더링합니다:


![](/images/reports/render_html.png)

  </TabItem>
</Tabs>

### 풍부한 미디어 링크 임베딩

App UI 또는 W&B SDK를 사용하여 리포트 내에 풍부한 미디어를 임베드하세요.

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

리포트 내에 풍부한 미디어를 임베딩하기 위해 URL을 복사하여 붙여넣으세요. 다음 애니메이션은 Twitter, YouTube 및 SoundCloud에서 URL을 복사하여 붙여넣는 방법을 보여줍니다

#### Twitter

리포트에 트윗 링크 URL을 복사하여 붙여넣어 리포트 내에서 트윗을 볼 수 있습니다.

![](/images/reports/twitter.gif)

####

#### Youtube

YouTube 비디오 URL 링크를 복사하여 붙여넣어 리포트에 비디오를 임베드하세요.

![](/images/reports/youtube.gif)

#### SoundCloud

SoundCloud 링크를 복사하여 붙여넣어 리포트에 오디오 파일을 임베드하세요.

![](/images/reports/soundcloud.gif)
  </TabItem>
  <TabItem value="sdk">

하나 이상의 임베디드 미디어 오브젝트를 `wandb.apis.reports.blocks` 속성에 목록으로 전달하세요. 다음 예제는 비디오와 Twitter 미디어를 리포트에 임베딩하는 방법을 보여줍니다:

```python
import wandb
import wandb.apis.reports as wr

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

### 패널 그리드 복제 및 삭제

재사용하고 싶은 레이아웃이 있다면, 패널 그리드를 선택하고 복사-붙여넣기하여 같은 리포트에서 복제하거나 다른 리포트에 붙여넣을 수 있습니다.

상단 오른쪽 모서리에서 드래그 핸들을 선택하여 전체 패널 그리드 섹션을 강조 표시합니다. 클릭하고 드래그하여 리포트의 영역(패널 그리드, 텍스트, 제목 등)을 강조 표시하고 선택하세요.

![](/images/reports/demo_copy_and_paste_a_panel_grid_section.gif)

키보드에서 `delete`를 눌러 패널 그리드를 삭제하세요.

![](@site/static/images/reports/delete_panel_grid.gif)

### 리포트를 정리하기 위해 헤더 축소

리포트에서 텍스트 블록 내의 내용을 숨기기 위해 헤더를 축소하세요. 리포트가 로드될 때, 확장된 헤더만 내용을 표시합니다. 리포트에서 헤더를 축소하면 내용을 정리하고 과도한 데이터 로딩을 방지할 수 있습니다. 다음 gif는 절차를 보여줍니다.

![](@site/static/images/reports/collapse_headers.gif)