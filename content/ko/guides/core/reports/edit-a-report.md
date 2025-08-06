---
title: 리포트 편집
description: App UI를 사용하여 리포트를 인터랙티브하게 편집하거나, W&B SDK로 프로그래밍 방식으로 편집할 수 있습니다.
menu:
  default:
    identifier: ko-guides-core-reports-edit-a-report
    parent: reports
weight: 20
---

{{% alert %}}
W&B Report 및 Workspace API는 Public Preview 단계입니다.
{{% /alert %}}

Report는 App UI에서 인터랙티브하게 또는 W&B SDK를 통해 프로그래밍 방식으로 편집할 수 있습니다.

Report는 _block_ 으로 구성됩니다. Block은 report의 본문을 구성하며, 각 block에는 텍스트, 이미지, 임베드된 시각화, 실험과 run에서 가져온 plot, 패널 grid 등을 추가할 수 있습니다.

_패널 grid_ 는 패널과 _run set_ 을 담는 특별한 종류의 block입니다. Run set은 W&B의 프로젝트에 기록된 run들의 집합입니다. 패널은 run set 데이터의 시각화입니다.


{{% alert %}}
[프로그램 방식 워크스페이스 튜토리얼]({{< relref path="/tutorials/workspaces.md" lang="ko" >}})에서 저장된 워크스페이스 뷰를 생성하고 커스터마이즈하는 과정을 단계별로 확인해 보세요.
{{% /alert %}}

{{% alert %}}
Report를 프로그래밍 방식으로 편집하려면, W&B Python SDK와 함께 W&B Report 및 Workspace API인 `wandb-workspaces`를 설치해야 합니다:

```pip
pip install wandb wandb-workspaces
```
{{% /alert %}}

## 플롯 추가하기

각 패널 grid는 run set과 패널의 집합을 가집니다. 이 섹션 하단의 run set이 grid 내 패널에 표시될 데이터를 제어합니다. 다른 run 집합의 데이터를 사용하는 차트를 추가하려면 새로운 패널 grid를 만드세요.

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

Report 내에서 슬래시(`/`)를 입력하면 드롭다운 메뉴가 표시됩니다. **Add panel**을 선택해서 패널을 추가하세요. W&B에서 지원하는 모든 패널(선형 그래프, 산점도, 패러럴 좌표 그래프 등)을 추가할 수 있습니다.

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="리포트에 차트 추가" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
SDK를 사용해 Report에 플롯을 프로그래밍 방식으로 추가할 수 있습니다. `PanelGrid` Public API 클래스의 `panels` 파라미터에 하나 이상의 plot 또는 chart 오브젝트 리스트를 전달하세요. 각 plot 또는 chart 오브젝트는 해당 Python 클래스로 생성합니다.

다음 예제는 선형 그래프(Line Plot)와 산점도(Scatter Plot)를 만드는 방법을 보여줍니다.

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

프로그램 방식으로 추가할 수 있는 plot과 chart에 대한 더 자세한 정보는 `wr.panels`를 참고하세요.

{{% /tab %}}
{{< /tabpane >}}


## Run set 추가하기

프로젝트에서 run set을 App UI 또는 W&B SDK를 이용해 인터랙티브하게 추가할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

Report 내에서 슬래시(`/`)를 입력해 드롭다운 메뉴를 띄우고, **Panel Grid**를 선택하세요. 이 작업은 report가 생성된 프로젝트로부터 run set을 자동으로 불러옵니다.

패널을 report에 임포트하면 run 이름은 프로젝트에서 상속됩니다. 필요하다면 report 내에서 [run 이름을 변경]({{< relref path="/guides/models/track/runs/#rename-a-run" lang="ko" >}})하여 독자가 더 많은 맥락을 얻을 수 있도록 할 수 있습니다. 이름 변경은 해당 패널에만 적용되며, 같은 report 내에서 패널을 복제할 경우 복제된 패널에도 이름이 반영됩니다.

1. Report에서 연필 아이콘을 클릭해 report editor를 엽니다.
1. Run set에서 이름을 변경하고 싶은 run을 찾습니다. run 이름 위에 마우스를 올리고 점 세 개 아이콘을 클릭합니다. 아래 옵션 중 하나를 선택하고 폼을 제출하세요.

    - **Rename run for project**: 프로젝트 전체에서 run의 이름을 변경합니다. 필드를 비워두면 새로운 임의의 이름이 생성됩니다.
    - **Rename run for panel grid**: report에서만 run의 이름을 변경하며, 다른 컨텍스트의 기존 이름은 그대로 둡니다. 새로운 임의의 이름 생성은 지원되지 않습니다.

1. **Publish report**를 클릭합니다.

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

`wr.Runset()` 및 `wr.PanelGrid` 클래스를 사용해 프로젝트에서 run set을 추가할 수 있습니다. 아래 과정은 runset을 추가하는 방법을 보여줍니다.

1. `wr.Runset()` 오브젝트 인스턴스를 생성하세요. 프로젝트 이름은 project 파라미터에, 프로젝트 소유자는 entity 파라미터에 전달하면 됩니다.
2. `wr.PanelGrid()` 오브젝트 인스턴스를 만들고, runset 오브젝트 리스트를 `run sets` 파라미터에 전달하세요.
3. 한 개 혹은 여러 개의 `wr.PanelGrid()` 오브젝트를 리스트에 저장하세요.
4. report 인스턴스의 blocks 속성을 panel grid 인스턴스 리스트로 업데이트합니다.

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

SDK의 한 번의 호출로 runset과 panel을 동시에 추가할 수도 있습니다:

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


## Run set 고정하기(Freeze)

Report는 자동으로 run set을 최신 상태로 업데이트하여, 프로젝트의 최신 데이터를 보여줍니다. 만약 특정 시점의 run set 상태를 고정하고 싶다면, 해당 run set을 *고정(Freeze)* 할 수 있습니다.

Report 화면에서 panel grid의 **Filter** 버튼 근처에 있는 눈송이 아이콘을 클릭하면 run set을 고정할 수 있습니다.

{{< img src="/images/reports/freeze_runset.png" alt="Freeze runset button" >}}

## Run set을 프로그래밍 방식으로 필터링하기

[Workspace 및 Reports API]({{< relref path="/ref/python/wandb_workspaces/reports" lang="ko" >}})를 사용해 run set을 프로그램 방식으로 필터링하여 report에 추가할 수 있습니다.

필터 표현식의 기본 문법은 다음과 같습니다:

```text
Filter('key') operation <value>
```

여기서 `key`는 필터 이름, `operation`은 비교 연산자(예: `>`, `<`, `==`, `in`, `not in`, `or`, `and`), `<value>`는 비교 대상 값입니다. `Filter`는 적용하려는 필터의 유형을 의미합니다. 사용 가능한 필터와 설명은 아래 표를 참고하세요:

| Filter | 설명 | 사용할 수 있는 key |
| ---|---| --- |
|`Config('key')` | config 값으로 필터 | `wandb.init(config=)`의 config 파라미터에 지정한 값 |
|`SummaryMetric('key')` | summary metric으로 필터 | `wandb.Run.log()`로 run에 기록한 값 |
|`Tags('key')` | 태그 값으로 필터 | 프로그램이나 W&B App에서 run에 추가한 태그 값 |
|`Metric('key')` | run 속성으로 필터 | `tags`, `state`, `displayName`, `jobType` |

필터를 정의한 후, `wr.PanelGrid(runsets=)`에 필터된 run set을 전달하여 report를 생성할 수 있습니다. 각 항목별로 report에 요소를 추가하는 방법은 이 페이지의 **Report and Workspace API** 탭을 참고하세요.

아래 예시에서는 report 내에서 run set을 어떻게 필터링하는지 보여줍니다.

### Config 필터

하나 이상의 config 값으로 runset을 필터링합니다. Config 값은 run의 설정(`wandb.init(config=)`)에 지정한 파라미터입니다.

예를 들어 아래 코드조각에서는 `learning_rate`와 `batch_size` config 값으로 run을 초기화한 후, `learning_rate`가 특정 값보다 큰 run만 report에서 필터링합니다.

```python
import wandb

config = {
    "learning_rate": 0.01,
    "batch_size": 32,
}

with wandb.init(project="<project>", entity="<entity>", config=config) as run:
    # 여기에 트레이닝 코드 입력
    pass
```

이후 Python 스크립트 또는 노트북 내에서 learning rate가 `0.01`보다 큰 run만 필터링할 수 있습니다.

```python
import wandb_workspaces.reports.v2 as wr

runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Config('learning_rate') > 0.01"
)
```

`and` 연산자를 사용해 여러 config 값으로 동시에 필터링할 수도 있습니다:
 
```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Config('learning_rate') > 0.01 and Config('batch_size') == 32"
)
```

이전 예시에서 필터링된 runset을 사용해 report를 만들 경우:

```python
report = wr.Report(
  entity="your_entity",
  project="your_project",
  title="My Report"
)

report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
      panels=[
          wr.LinePlot(
              x="Step",
              y=["accuracy"],
          )
      ]
  )
]

report.save()
```

### Metric 필터

Run의: 태그(`tags`), run 상태(`state`), run 이름(`displayName`), job 유형(`jobType`) 값으로 run set을 필터링할 수 있습니다.

{{% alert %}}
`Metric` 필터는 다른 문법을 사용합니다. 값 리스트를 list 형태로 전달하세요.

```text
Metric('key') operation [<value>]
```
{{% /alert %}}

예를 들어, 다음 Python 코드조각은 세 개의 run을 생성하고 각각에 이름을 할당합니다:

```python
import wandb

with wandb.init(project="<project>", entity="<entity>") as run:
    for i in range(3):
        run.name = f"run{i+1}"
        # 여기에 트레이닝 코드 입력
        pass
```

Report를 만들 때 run의 display name으로 필터링할 수 있습니다. 예를 들어 이름이 `run1`, `run2`, `run3`인 run만 필터링하려면 다음과 같이 쓸 수 있습니다:

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Metric('displayName') in ['run1', 'run2', 'run3']"
)
```

{{% alert %}}
Run의 이름은 W&B App의 run **Overview** 페이지에서 확인하거나, 프로그램 방식으로는 `Api.runs().run.name`을 사용해 알 수 있습니다.
{{% /alert %}}

다음 예제는 run의 상태(`finished`, `crashed`, `running` 등)로 runset을 필터링하는 방법을 보여줍니다:

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Metric('state') in ['finished']"
)
```

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Metric('state') not in ['crashed']"
)
```

### SummaryMetric 필터

Summary metric으로 runset을 필터링하는 방법은 아래와 같습니다. Summary metric은 `wandb.Run.log()`로 run에 기록한 값입니다. log 후에는 W&B App의 run **Overview** > **Summary** 섹션에서 metric 이름을 확인할 수 있습니다.

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="SummaryMetric('accuracy') > 0.9"
)
```

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Metric('state') in ['finished'] and SummaryMetric('train/train_loss') < 0.5"
)
```

### Tags 필터

아래 코드조각은 run set을 태그 값으로 필터링하는 방법을 보여줍니다. 태그는 run에 프로그램 조건 또는 W&B App에서 추가할 수 있습니다.

```python
runset = wr.Runset(
  entity="your_entity",
  project="your_project",
  filters="Tags('training') == 'training'"
)
```

## 코드 블록 추가하기

코드 블록은 App UI 또는 W&B SDK를 사용해 report에 추가할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

Report 내에서 슬래시(`/`)를 입력해 드롭다운 메뉴를 띄우고, **Code**를 선택하세요.

코드 블록 우측에서 프로그래밍 언어를 선택하면 드롭다운이 펼쳐집니다. 원하는 프로그래밍 언어 문법을 선택할 수 있습니다. Javascript, Python, CSS, JSON, HTML, Markdown, YAML 등을 지원합니다.

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wr.CodeBlock` 클래스를 사용해 코드 블록을 프로그램 방식으로 생성할 수 있습니다. language와 code 파라미터에 각각 언어 이름과 표시할 코드를 전달하세요.

아래 예시는 YAML 파일의 리스트를 보여줍니다:

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

위 코드는 다음과 같은 코드 블록으로 렌더링됩니다:

```yaml
this:
- is
- a
cool:
- yaml
- file
```

다음 예시는 Python 코드 블록을 보여줍니다:

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

아래와 유사한 코드 블록이 생성됩니다:

```md
Hello, World!
```

{{% /tab %}}

{{% /tabpane %}}

## 마크다운 추가하기

마크다운은 App UI 또는 W&B SDK를 사용해 report에 추가할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

Report 내에서 슬래시(`/`)를 입력해 드롭다운 메뉴를 띄운 후 **Markdown**을 선택하세요.

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

`wandb.apis.reports.MarkdownBlock` 클래스를 사용해 프로그램 방식으로 마크다운 블록을 생성할 수 있습니다. `text` 파라미터에 문자열을 전달하세요:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

아래와 같이 마크다운 블록이 렌더링됩니다:

{{< img src="/images/reports/markdown.png" alt="Rendered markdown block" >}}

{{% /tab %}}

{{% /tabpane %}}


## HTML 요소 추가하기

HTML 요소는 App UI 또는 W&B SDK를 통해 report에 추가할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

Report 내에서 슬래시(`/`)를 입력해 드롭다운 메뉴를 띄운 후 원하는 텍스트 블록 유형을 선택하세요. 예를 들어 H2 헤딩 블록을 생성하려면 `Heading 2` 옵션을 선택하세요.

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

HTML 요소 리스트를 `wandb.apis.reports.blocks` 속성에 전달해 추가할 수 있습니다. 아래 예시는 H1, H2, 그리고 비순차 리스트 생성 예시입니다:

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

위 코드는 다음과 같이 HTML 요소로 렌더링됩니다:


{{< img src="/images/reports/render_html.png" alt="Rendered HTML elements" >}}

{{% /tab %}}

{{% /tabpane %}}

## 리치 미디어 링크 임베드하기

리포트 내에 리치 미디어를 App UI 또는 W&B SDK를 사용해 임베드할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

URL을 복사해서 report에 붙여넣으면 리치 미디어가 리포트 내에 임베드됩니다. 아래 애니메이션은 Twitter, YouTube, SoundCloud에서 URL을 복사/붙여넣는 방법을 보여줍니다.

### Twitter

트윗의 링크 URL을 복사해 report에 붙여넣으면 리포트 내에서 해당 트윗을 볼 수 있습니다.

{{< img src="/images/reports/twitter.gif" alt="Embedding Twitter content" >}}

### Youtube

YouTube 영상 링크를 복사해서 report에 붙여넣으면 영상이 임베드됩니다.

{{< img src="/images/reports/youtube.gif" alt="Embedding YouTube videos" >}}

### SoundCloud

SoundCloud 링크를 복사해 report에 붙여넣으면 오디오 파일을 임베드할 수 있습니다.

{{< img src="/images/reports/soundcloud.gif" alt="Embedding SoundCloud audio" >}}

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

하나 이상의 임베드 미디어 오브젝트를 `wandb.apis.reports.blocks` 속성에 리스트 형태로 전달하세요. 아래 예시는 report에 비디오와 트위터 미디어를 임베드하는 방법을 보여줍니다:

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

## 패널 grid 복제 및 삭제하기

레이아웃을 재사용하고 싶다면, 패널 grid를 선택 후 복사해 같은 report에 붙여넣거나 다른 report에 붙여넣을 수 있습니다.

패널 grid 영역 전체를 선택하려면 우측 상단의 드래그 핸들을 선택하세요. 클릭 후 드래그로 report 내 패널 grid, 텍스트, 헤딩 등 원하는 영역을 선택할 수 있습니다.

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="Copying panel grids" >}}

패널 grid를 선택하고 키보드의 `delete`를 누르면 해당 panel grid가 삭제됩니다.

{{< img src="/images/reports/delete_panel_grid.gif" alt="Deleting panel grids" >}}

## Report 내에서 헤더 접기로 리포트 정리하기

Report의 헤더를 접어(header collapse) 텍스트 블록 내 콘텐츠를 숨길 수 있습니다. report를 불러올 때 펼쳐진 헤더에만 콘텐츠가 표시됩니다. Report에서 헤더를 접는 기능은 콘텐츠 정리와 불필요한 데이터 로딩 방지에 유용합니다. 아래 gif에서 과정을 볼 수 있습니다.

{{< img src="/images/reports/collapse_headers.gif" alt="Collapsing headers in a report." >}}

## 다차원 관계 시각화하기

여러 차원의 관계를 효과적으로 시각화하려면 변수 중 한 개를 색상 그레이디언트로 표현하세요. 이는 해석의 명확성을 높이고 패턴 이해를 돕습니다.

1. 색상 그레이디언트로 표현할 변수를 선택하세요(예: penalty 점수, learning rate 등). 이를 통해 penalty(색상)와 reward/side effect(y축)가 트레이닝 시간(x축)에 따라 어떻게 상호작용하는지 더 명확히 파악할 수 있습니다.
2. 주요 트렌드를 강조하세요. 특정 run 그룹에 마우스를 올리면 해당 그룹이 시각화에서 강조 표기됩니다.