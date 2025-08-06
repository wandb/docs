---
title: Reports
menu:
  reference:
    identifier: ko-ref-python-wandb_workspaces-reports
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py" >}}




{{% alert %}}
W&B Report 및 Workspace API는 Public Preview 단계에 있습니다.
{{% /alert %}}

# <kbd>module</kbd> `wandb_workspaces.reports.v2`
W&B Reports API를 프로그래밍적으로 다루기 위한 Python 라이브러리입니다.

```python
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
     entity="entity",
     project="project",
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

---

## <kbd>class</kbd> `BarPlot`
2D 바 플롯을 보여주는 패널 오브젝트입니다.

**속성:**
 
 - `title` (Optional[str]): 플롯 상단에 표시될 텍스트입니다.
 - `metrics` (LList[MetricType]): orientation Literal["v", "h"]: 바 플롯의 방향을 지정합니다. 세로("v") 또는 가로("h")로 설정할 수 있으며, 기본값은 가로("h")입니다.
 - `range_x` (Tuple[float | None, float | None]): x축의 범위를 지정하는 튜플입니다.
 - `title_x` (Optional[str]): x축의 라벨입니다.
 - `title_y` (Optional[str]): y축의 라벨입니다.
 - `groupby` (Optional[str]): Report에서 정보를 가져올 때 기준이 되는 W&B 프로젝트에 로그된 메트릭으로 run들을 그룹화합니다.
 - `groupby_aggfunc` (Optional[GroupAgg]): 지정한 함수로 run들을 집계합니다. 사용 가능한 옵션: `mean`, `min`, `max`, `median`, `sum`, `samples`, 또는 `None`.
 - `groupby_rangefunc` (Optional[GroupArea]): 범위에 따라 run들을 그룹화합니다. 사용 가능한 옵션: `minmax`, `stddev`, `stderr`, `none`, `samples`, 또는 `None`.
 - `max_runs_to_show` (Optional[int]): 플롯에 표시할 최대 run 개수입니다.
 - `max_bars_to_show` (Optional[int]): 바 플롯에 표시할 최대 바 개수입니다.
 - `custom_expressions` (Optional[LList[str]]): 바 플롯에 사용할 커스텀 표현식의 리스트입니다.
 - `legend_template` (Optional[str]): 범례의 템플릿입니다.
 - `font_size` ( Optional[FontSize]): 라인 플롯 폰트의 크기입니다. 사용 가능한 옵션: `small`, `medium`, `large`, `auto`, 또는 `None`.
 - `line_titles` (Optional[dict]): 각 라인의 제목입니다. 키는 라인 이름이고 값은 제목입니다.
 - `line_colors` (Optional[dict]): 각 라인의 색상입니다. 키는 라인 이름이고 값은 색상입니다.

---

## <kbd>class</kbd> `BlockQuote`
인용문 텍스트 블록입니다.

**속성:**
 
 - `text` (str): 인용 블록의 텍스트입니다.

---

## <kbd>class</kbd> `CalloutBlock`
주목할 만한 텍스트 블록입니다.

**속성:**
 
 - `text` (str): 강조 텍스트입니다.

---

## <kbd>class</kbd> `CheckedList`
체크박스가 있는 항목 리스트입니다. 하나 이상의 `CheckedListItem`을 `CheckedList`에 포함시킵니다.

**속성:**
 
 - `items` (LList[CheckedListItem]): 하나 이상의 `CheckedListItem` 오브젝트의 리스트입니다.

---

## <kbd>class</kbd> `CheckedListItem`
체크박스가 포함된 리스트 항목입니다. 하나 이상의 `CheckedListItem`을 `CheckedList`에 추가합니다.

**속성:**
 
 - `text` (str): 리스트 항목의 텍스트입니다.
 - `checked` (bool): 체크박스의 체크 여부입니다. 기본값은 `False`입니다.

---

## <kbd>class</kbd> `CodeBlock`
코드 블록입니다.

**속성:**
 
 - `code` (str): 블록 내의 코드입니다.
 - `language` (Optional[Language]): 코드의 언어입니다. 지정된 언어는 문법 하이라이팅에 사용되며, 기본값은 `python`입니다. 사용 가능한 옵션: `javascript`, `python`, `css`, `json`, `html`, `markdown`, `yaml`.

---

## <kbd>class</kbd> `CodeComparer`
서로 다른 두 run의 코드 비교를 위한 패널 오브젝트입니다.

**속성:**
 
 - `diff` `(Literal['split', 'unified'])`: 코드 차이의 표시 방식입니다. `split` 및 `unified` 옵션을 사용할 수 있습니다.

---

## <kbd>class</kbd> `Config`
run의 config 오브젝트에 로그된 메트릭입니다. 일반적으로 `wandb.Run.config[name] = ...` 형식 또는 키-값 쌍의 딕셔너리로 config를 전달하여 기록합니다. 여기서 키는 메트릭의 이름, 값은 그 메트릭의 값입니다.

**속성:**
 
 - `name` (str): 메트릭의 이름입니다.

---

## <kbd>class</kbd> `CustomChart`
커스텀 차트를 보여주는 패널입니다. 차트는 weave 쿼리로 정의됩니다.

**속성:**
 
 - `query` (dict): 커스텀 차트를 정의하는 쿼리입니다. 키는 필드명, 값은 쿼리입니다.
 - `chart_name` (str): 커스텀 차트의 제목입니다.
 - `chart_fields` (dict): 플롯의 축을 정의하는 키-값 쌍입니다. 키가 라벨, 값이 메트릭입니다.
 - `chart_strings` (dict): 차트 내 문자열을 정의하는 키-값 쌍입니다.

---

### <kbd>classmethod</kbd> `from_table`

```python
from_table(
    table_name: str,
    chart_fields: dict = None,
    chart_strings: dict = None
)
```

테이블에서 커스텀 차트를 생성합니다.

**인수:**
 
 - `table_name` (str): 테이블 이름입니다.
 - `chart_fields` (dict): 차트에 표시할 필드입니다.
 - `chart_strings` (dict): 차트에 표시할 문자열입니다.

---

## <kbd>class</kbd> `Gallery`
reports 및 URL들의 갤러리를 렌더링하는 블록입니다.

**속성:**
 
 - `items` (List[Union[`GalleryReport`, `GalleryURL`]]): `GalleryReport` 및 `GalleryURL` 오브젝트의 리스트입니다.

---

## <kbd>class</kbd> `GalleryReport`
갤러리 내 report를 참조하는 오브젝트입니다.

**속성:**
 
 - `report_id` (str): report의 ID입니다.

---

## <kbd>class</kbd> `GalleryURL`
외부 리소스에 대한 URL입니다.

**속성:**
 
 - `url` (str): 리소스의 URL입니다.
 - `title` (Optional[str]): 리소스의 제목입니다.
 - `description` (Optional[str]): 리소스의 설명입니다.
 - `image_url` (Optional[str]): 표시할 이미지의 URL입니다.

---

## <kbd>class</kbd> `GradientPoint`
그레이디언트 내의 포인트입니다.

**속성:**
 
 - `color`: 포인트의 색상입니다.
 - `offset`: 그레이디언트 내 포인트의 위치입니다. 0과 100 사이의 값이어야 합니다.

---

## <kbd>class</kbd> `H1`
지정된 텍스트로 H1 헤딩을 생성합니다.

**속성:**
 
 - `text` (str): 헤딩에 표시될 텍스트입니다.
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 헤딩이 축소되었을 때 표시할 블록입니다.

---

## <kbd>class</kbd> `H2`
지정된 텍스트로 H2 헤딩을 생성합니다.

**속성:**
 
 - `text` (str): 헤딩에 표시될 텍스트입니다.
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 헤딩이 축소되었을 때 하나 이상의 블록을 표시합니다.

---

## <kbd>class</kbd> `H3`
지정된 텍스트로 H3 헤딩을 생성합니다.

**속성:**
 
 - `text` (str): 헤딩에 표시될 텍스트입니다.
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): 헤딩이 축소되었을 때 하나 이상의 블록을 표시합니다.

---

## <kbd>class</kbd> `Heading`

---

## <kbd>class</kbd> `HorizontalRule`
HTML 수평선입니다.

---

## <kbd>class</kbd> `Image`
이미지를 렌더링하는 블록입니다.

**속성:**
 
 - `url` (str): 이미지의 URL입니다.
 - `caption` (str): 이미지 아래에 표시되는 캡션입니다.

---

## <kbd>class</kbd> `InlineCode`
인라인 코드 블록입니다. 코드 뒤에 줄바꿈 문자를 추가하지 않습니다.

**속성:**
 
 - `text` (str): 리포트에 표시할 코드입니다.

---

## <kbd>class</kbd> `InlineLatex`
인라인 LaTeX 마크다운입니다. LaTeX 마크다운 뒤에 줄바꿈 문자를 추가하지 않습니다.

**속성:**
 
 - `text` (str): 리포트에 표시될 LaTeX 마크다운입니다.

---

## <kbd>class</kbd> `LatexBlock`
LaTeX 텍스트 블록입니다.

**속성:**
 
 - `text` (str): LaTeX 텍스트입니다.

---

## <kbd>class</kbd> `Layout`
리포트 내 패널의 레이아웃입니다. 패널의 크기와 위치를 조정합니다.

**속성:**
 
 - `x` (int): 패널의 x 위치입니다.
 - `y` (int): 패널의 y 위치입니다.
 - `w` (int): 패널의 너비입니다.
 - `h` (int): 패널의 높이입니다.

---

## <kbd>class</kbd> `LinePlot`
2D 라인 플롯을 제공하는 패널 오브젝트입니다.

**속성:**
 
 - `title` (Optional[str]): 플롯 상단에 표시될 텍스트입니다.
 - `x` (Optional[MetricType]): 정보를 불러올 때 x축에 사용할 W&B 프로젝트에 로그된 메트릭입니다.
 - `y` (LList[MetricType]): 정보를 불러올 때 y축에 사용할 하나 이상의 W&B 메트릭입니다.
 - `range_x` (Tuple[float | `None`, float | `None`]): x축의 범위를 지정하는 튜플입니다.
 - `range_y` (Tuple[float | `None`, float | `None`]): y축의 범위를 지정하는 튜플입니다.
 - `log_x` (Optional[bool]): x좌표를 base-10 로그 스케일로 플롯합니다.
 - `log_y` (Optional[bool]): y좌표를 base-10 로그 스케일로 플롯합니다.
 - `title_x` (Optional[str]): x축의 라벨입니다.
 - `title_y` (Optional[str]): y축의 라벨입니다.
 - `ignore_outliers` (Optional[bool]): `True`로 설정하면 이상치를 플롯하지 않습니다.
 - `groupby` (Optional[str]): 정보 조회에 사용하는 메트릭으로 run을 그룹화합니다.
 - `groupby_aggfunc` (Optional[GroupAgg]): 지정한 함수로 run들을 집계합니다. 옵션: `mean`, `min`, `max`, `median`, `sum`, `samples`, `None`.
 - `groupby_rangefunc` (Optional[GroupArea]): 지정한 범위별로 run을 그룹화합니다. 옵션: `minmax`, `stddev`, `stderr`, `none`, `samples`, `None`.
 - `smoothing_factor` (Optional[float]): 스무딩에 적용할 팩터입니다. 허용 범위는 0~1입니다.
 - `smoothing_type Optional[SmoothingType]`: 지정한 분포 기반의 필터를 적용합니다. 사용 가능한 옵션은 `exponentialTimeWeighted`, `exponential`, `gaussian`, `average`, `none`입니다.
 - `smoothing_show_original` (Optional[bool]): `True`로 설정하면 원본 데이터를 표시합니다.
 - `max_runs_to_show` (Optional[int]): 라인 플롯에 표시할 최대 run 개수입니다.
 - `custom_expressions` (Optional[LList[str]]): 데이터에 적용할 커스텀 표현식입니다.
 - `plot_type Optional[LinePlotStyle]`: 생성할 라인 플롯 타입을 지정합니다. 사용 가능한 옵션: `line`, `stacked-area`, `pct-area`.
 - `font_size Optional[FontSize]`: 라인 플롯 폰트 크기입니다. `small`, `medium`, `large`, `auto`, `None` 중 선택 가능합니다.
 - `legend_position Optional[LegendPosition]`: 범례의 위치를 지정합니다. 옵션: `north`, `south`, `east`, `west`, `None`.
 - `legend_template` (Optional[str]): 범례 템플릿입니다.
 - `aggregate` (Optional[bool]): `True`로 설정하면 데이터를 집계합니다.
 - `xaxis_expression` (Optional[str]): x축 표현식입니다.
 - `legend_fields` (Optional[LList[str]]): 범례에 포함할 필드입니다.

---

## <kbd>class</kbd> `Link`
URL에 대한 링크입니다.

**속성:**
 
 - `text` (Union[str, TextWithInlineComments]): 링크에 표시될 텍스트입니다.
 - `url` (str): 링크가 가리키는 URL입니다.

---

## <kbd>class</kbd> `MarkdownBlock`
마크다운 텍스트 블록입니다. 마크다운 문법을 사용하는 텍스트를 작성하고 싶을 때 유용합니다.

**속성:**
 
 - `text` (str): 마크다운 텍스트입니다.

---

## <kbd>class</kbd> `MarkdownPanel`
마크다운을 렌더링하는 패널입니다.

**속성:**
 
 - `markdown` (str): 마크다운 패널에 표시할 텍스트입니다.

---

## <kbd>class</kbd> `MediaBrowser`
미디어 파일을 그리드 레이아웃으로 표시하는 패널입니다.

**속성:**
 
 - `num_columns` (Optional[int]): 그리드 컬럼 개수입니다.
 - `media_keys` (LList[str]): 미디어 파일에 해당하는 미디어 키의 리스트입니다.

---

## <kbd>class</kbd> `Metric`
프로젝트에 기록된 메트릭 중, 리포트에 표시할 메트릭입니다.

**속성:**
 
 - `name` (str): 메트릭의 이름입니다.

---

## <kbd>class</kbd> `OrderBy`
정렬 기준이 되는 메트릭입니다.

**속성:**
 
 - `name` (str): 메트릭의 이름입니다.
 - `ascending` (bool): 오름차순 정렬 여부입니다. 기본값은 `False`입니다.

---

## <kbd>class</kbd> `OrderedList`
숫자 리스트로 이루어진 항목 리스트입니다.

**속성:**
 
 - `items` (LList[str]): 하나 이상의 `OrderedListItem` 오브젝트의 리스트입니다.

---

## <kbd>class</kbd> `OrderedListItem`
숫자 리스트 내 항목입니다.

**속성:**
 
 - `text` (str): 리스트 항목의 텍스트입니다.

---

## <kbd>class</kbd> `P`
문단 텍스트입니다.

**속성:**
 
 - `text` (str): 문단의 텍스트입니다.

---

## <kbd>class</kbd> `Panel`
패널 그리드에서 시각화를 표시하는 패널입니다.

**속성:**
 
 - `layout` (Layout): `Layout` 오브젝트입니다.

---

## <kbd>class</kbd> `PanelGrid`
runset과 패널로 구성된 그리드입니다. 각각 `Runset` 및 `Panel` 오브젝트를 추가하여 구성합니다.

사용 가능한 패널: `LinePlot`, `ScatterPlot`, `BarPlot`, `ScalarChart`, `CodeComparer`, `ParallelCoordinatesPlot`, `ParameterImportancePlot`, `RunComparer`, `MediaBrowser`, `MarkdownPanel`, `CustomChart`, `WeavePanel`, `WeavePanelSummaryTable`, `WeavePanelArtifactVersionedFile`.

**속성:**
 
 - `runsets` (LList["Runset"]): 하나 이상의 `Runset` 오브젝트를 담은 리스트입니다.
 - `panels` (LList["PanelTypes"]): 하나 이상의 `Panel` 오브젝트를 담은 리스트입니다.
 - `active_runset` (int): runset 내에 표시할 run 수입니다. 기본값은 0입니다.
 - `custom_run_colors` (dict): 키는 run의 이름이고 값은 16진수 색상값으로 지정한 컬러입니다.

---

## <kbd>class</kbd> `ParallelCoordinatesPlot`
병렬 좌표 플롯을 보여주는 패널 오브젝트입니다.

**속성:**
 
 - `columns` (LList[ParallelCoordinatesPlotColumn]): 하나 이상의 `ParallelCoordinatesPlotColumn` 오브젝트 리스트입니다.
 - `title` (Optional[str]): 플롯 상단에 표시될 텍스트입니다.
 - `gradient` (Optional[LList[GradientPoint]]): 그레이디언트 포인트 리스트입니다.
 - `font_size` (Optional[FontSize]): 라인 플롯 폰트 크기입니다. `small`, `medium`, `large`, `auto`, `None` 중 선택 가능합니다.

---

## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
Parallel coordinates plot 내 컬럼입니다. 지정된 `metric`의 순서가 x축(병렬축)의 순서를 결정합니다.

**속성:**
 
 - `metric` (str | Config | SummaryMetric): Report에서 정보를 조회할 때 기준이 되는 W&B 메트릭 이름입니다.
 - `display_name` (Optional[str]): 메트릭의 표시 이름입니다.
 - `inverted` (Optional[bool]): 메트릭 반전 여부입니다.
 - `log` (Optional[bool]): 메트릭에 로그 변환 적용 여부입니다.

---

## <kbd>class</kbd> `ParameterImportancePlot`
선택한 메트릭을 예측하는 데 있어 각각의 하이퍼파라미터가 얼마나 중요한지를 보여주는 패널입니다.

**속성:**
 
 - `with_respect_to` (str): 파라미터 중요도를 비교할 메트릭입니다. 보통 loss, accuracy 등의 메트릭 값이 사용됩니다. 명시한 메트릭은 report가 정보를 불러오는 프로젝트 내에 로그되어 있어야 합니다.

---

## <kbd>class</kbd> `Report`
W&B Report를 나타내는 오브젝트입니다. 반환된 오브젝트의 `blocks` 속성을 이용해 리포트 내용을 커스터마이즈할 수 있습니다. Report 오브젝트는 자동으로 저장되지 않으므로, 변경 사항은 `save()` 메소드로 저장해야 합니다.

**속성:**
 
 - `project` (str): 리포트에 불러올 W&B 프로젝트의 이름입니다. report의 URL에서 확인할 수 있습니다.
 - `entity` (str): 리포트를 소유한 W&B entity입니다. report의 URL에서 볼 수 있습니다.
 - `title` (str): 리포트의 제목입니다. H1 헤딩으로 report 상단에 표시됩니다.
 - `description` (str): 리포트의 설명입니다. 제목 아래에 나타납니다.
 - `blocks` (LList[BlockTypes]): 하나 이상의 HTML 태그, 플롯, 그리드, runset 등으로 구성된 리스트입니다.
 - `width` (Literal['readable', 'fixed', 'fluid']): 리포트의 너비입니다. 'readable', 'fixed', 'fluid' 옵션 중 선택할 수 있습니다.

---

#### <kbd>property</kbd> url

report가 호스팅되는 URL입니다. URL 형식은 `https://wandb.ai/{entity}/{project_name}/reports/`로, `{entity}`와 `{project_name}`은 각각 report가 소속된 entity와 프로젝트명을 의미합니다.

---

### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str, as_model: bool = False)
```

지정한 URL에서 report를 현재 환경으로 불러옵니다.

**인수:**
 
 - `url` (str): report가 호스팅되는 URL입니다.
 - `as_model` (bool): True로 설정하면 Report 오브젝트가 아닌 모델 오브젝트를 반환합니다. 기본값은 `False`입니다.

---

### <kbd>method</kbd> `save`

```python
save(draft: bool = False, clone: bool = False)
```

Report 오브젝트에 적용한 변경사항을 저장합니다.

---

### <kbd>method</kbd> `to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

해당 report를 표시하는 iframe이 포함된 HTML을 생성합니다. 주로 Python 노트북에서 사용합니다.

**인수:**
 
 - `height` (int): iframe의 높이입니다.
 - `hidden` (bool): True로 설정하면 iframe을 숨깁니다. 기본값은 `False`입니다.

---

## <kbd>class</kbd> `RunComparer`
Report에서 정보를 불러오는 프로젝트 내에서 서로 다른 run의 메트릭을 비교하는 패널입니다.

**속성:**
 
 - `diff_only` `(Optional[Literal["split", True]])`: 프로젝트 내에서 run 간의 차이만 표시합니다. 이 기능은 W&B Report UI에서 토글할 수 있습니다.

---

## <kbd>class</kbd> `Runset`
패널 그리드에 표시할 run의 집합입니다.

**속성:**
 
 - `entity` (str): run이 저장된 프로젝트를 소유하거나 관련 권한이 있는 entity입니다.
 - `project` (str): run이 저장된 프로젝트의 이름입니다.
 - `name` (str): run set의 이름입니다. 기본값은 `Run set`입니다.
 - `query` (str): run을 필터링하기 위한 쿼리 스트링입니다.
 - `filters` (Optional[str]): run을 필터링하기 위한 필터 문자열입니다.
 - `groupby` (LList[str]): 그룹화를 위한 메트릭 이름의 리스트입니다.
 - `order` (LList[OrderBy]): 정렬 기준이 되는 `OrderBy` 오브젝트의 리스트입니다.
 - `custom_run_colors` (LList[OrderBy]): run ID별로 색상을 매핑하는 딕셔너리입니다.

---

## <kbd>class</kbd> `RunsetGroup`
runset들의 그룹을 보여주는 UI 요소입니다.

**속성:**
 
 - `runset_name` (str): runset의 이름입니다.
 - `keys` (Tuple[RunsetGroupKey, ...]): 그룹화할 때 사용할 키입니다. 하나 이상의 `RunsetGroupKey` 오브젝트를 전달합니다.

---

## <kbd>class</kbd> `RunsetGroupKey`
메트릭 타입과 값에 따라 runset을 그룹화합니다. `RunsetGroup`의 일부로, 그룹화할 메트릭 타입과 값을 키-값 쌍으로 지정합니다.

**속성:**
 
 - `key` (Type[str] | Type[Config] | Type[SummaryMetric] | Type[Metric]): 그룹화할 메트릭 타입입니다.
 - `value` (str): 그룹화할 값입니다.

---

## <kbd>class</kbd> `ScalarChart`
스칼라 차트를 보여주는 패널 오브젝트입니다.

**속성:**
 
 - `title` (Optional[str]): 플롯 상단에 표시될 텍스트입니다.
 - `metric` (MetricType): Report에서 정보를 불러올 때 사용할 메트릭 이름입니다.
 - `groupby_aggfunc` (Optional[GroupAgg]): 지정된 함수로 run을 집계합니다. 사용 가능한 옵션: `mean`, `min`, `max`, `median`, `sum`, `samples`, 또는 `None`.
 - `groupby_rangefunc` (Optional[GroupArea]): 범위로 run을 그룹화합니다. 옵션: `minmax`, `stddev`, `stderr`, `none`, `samples`, `None`.
 - `custom_expressions` (Optional[LList[str]]): 스칼라 차트에 사용할 커스텀 표현식 리스트입니다.
 - `legend_template` (Optional[str]): 범례 템플릿입니다.
 - `font_size Optional[FontSize]`: 라인 플롯 폰트 크기입니다. `small`, `medium`, `large`, `auto`, `None` 중 선택할 수 있습니다.

---

## <kbd>class</kbd> `ScatterPlot`
2D 또는 3D 산점도를 보여주는 패널 오브젝트입니다.

**인수:**
 
 - `title` (Optional[str]): 플롯 상단에 표시될 텍스트입니다.
 - `x Optional[SummaryOrConfigOnlyMetric]`: x축에 사용할 W&B 프로젝트 내 로그된 메트릭의 이름입니다.
 - `y Optional[SummaryOrConfigOnlyMetric]`: y축에 사용할 하나 이상의 W&B 프로젝트 내 로그된 메트릭입니다. z Optional[SummaryOrConfigOnlyMetric]:
 - `range_x` (Tuple[float | `None`, float | `None`]): x축 범위를 지정하는 튜플입니다.
 - `range_y` (Tuple[float | `None`, float | `None`]): y축 범위를 지정하는 튜플입니다.
 - `range_z` (Tuple[float | `None`, float | `None`]): z축 범위를 지정하는 튜플입니다.
 - `log_x` (Optional[bool]): x좌표를 base-10 로그 스케일로 플롯합니다.
 - `log_y` (Optional[bool]): y좌표를 base-10 로그 스케일로 플롯합니다.
 - `log_z` (Optional[bool]): z좌표를 base-10 로그 스케일로 플롯합니다.
 - `running_ymin` (Optional[bool]): 이동 평균 또는 롤링 평균을 적용합니다.
 - `running_ymax` (Optional[bool]): 이동 평균 또는 롤링 평균을 적용합니다.
 - `running_ymean` (Optional[bool]): 이동 평균 또는 롤링 평균을 적용합니다.
 - `legend_template` (Optional[str]): 범례의 형식을 지정하는 문자열입니다.
 - `gradient` (Optional[LList[GradientPoint]]): 플롯의 색상 그레이디언트를 지정하는 그레이디언트 포인트 리스트입니다.
 - `font_size` (Optional[FontSize]): 라인 플롯 폰트 크기입니다. `small`, `medium`, `large`, `auto`, `None` 중 선택 가능합니다.
 - `regression` (Optional[bool]): `True`로 설정하면 산점도에 회귀선을 그립니다.

---

## <kbd>class</kbd> `SoundCloud`
SoundCloud 플레이어를 렌더링하는 블록입니다.

**속성:**
 
 - `html` (str): SoundCloud 플레이어를 삽입할 HTML 코드입니다.

---

## <kbd>class</kbd> `Spotify`
Spotify 플레이어를 렌더링하는 블록입니다.

**속성:**
 
 - `spotify_id` (str): 트랙 또는 플레이리스트의 Spotify ID입니다.

---

## <kbd>class</kbd> `SummaryMetric`
리포트에 표시할 summary 메트릭입니다.

**속성:**
 
 - `name` (str): 메트릭의 이름입니다.

---

## <kbd>class</kbd> `TableOfContents`
리포트에 지정된 H1, H2, H3 HTML 블록으로 구성된 섹션 및 하위 섹션 목록을 포함하는 블록입니다.

---

## <kbd>class</kbd> `TextWithInlineComments`
인라인 코멘트가 포함된 텍스트 블록입니다.

**속성:**
 
 - `text` (str): 블록의 텍스트입니다.

---

## <kbd>class</kbd> `Twitter`
Twitter 피드를 표시하는 블록입니다.

**속성:**
 
 - `html` (str): Twitter 피드를 표시할 HTML 코드입니다.

---

## <kbd>class</kbd> `UnorderedList`
점으로 표시되는 리스트 항목의 리스트입니다.

**속성:**
 
 - `items` (LList[str]): 하나 이상의 `UnorderedListItem` 오브젝트의 리스트입니다.

---

## <kbd>class</kbd> `UnorderedListItem`
점 리스트 내 항목입니다.

**속성:**
 
 - `text` (str): 리스트 항목의 텍스트입니다.

---

## <kbd>class</kbd> `Video`
비디오를 렌더링하는 블록입니다.

**속성:**
 
 - `url` (str): 비디오의 URL입니다.

---

## <kbd>class</kbd> `WeaveBlockArtifact`
W&B에 기록된 artifact를 보여주는 블록입니다. 쿼리 형식은 다음과 같습니다.

```python
project('entity', 'project').artifact('artifact-name')
```

API 이름의 "Weave"는 LLM 트래킹 및 평가를 위한 W&B Weave 툴킷을 의미하지 않습니다.

**속성:**
 
 - `entity` (str): artifact가 저장된 프로젝트에 적절한 권한을 가진 entity입니다.
 - `project` (str): artifact가 저장된 프로젝트입니다.
 - `artifact` (str): 가져올 artifact의 이름입니다.
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: artifact 패널에서 표시할 탭입니다.

---

## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`
W&B artifact에 기록된 버전 파일을 보여주는 블록입니다. 쿼리 형식은 다음과 같습니다.

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 이름의 "Weave"는 LLM 트래킹 및 평가를 위한 W&B Weave 툴킷을 의미하지 않습니다.

**속성:**
 
 - `entity` (str): artifact가 저장된 프로젝트에 적절한 권한을 가진 entity입니다.
 - `project` (str): artifact가 저장된 프로젝트입니다.
 - `artifact` (str): 가져올 artifact의 이름입니다.
 - `version` (str): 가져올 artifact의 버전입니다.
 - `file` (str): 가져올 artifact 내 파일 이름입니다.

---

## <kbd>class</kbd> `WeaveBlockSummaryTable`
W&B Table, pandas DataFrame, 플롯 혹은 W&B에 기록된 기타 값을 보여주는 블록입니다. 쿼리 형식은 다음과 같습니다.

```python
project('entity', 'project').runs.summary['value']
```

API 이름의 "Weave"는 LLM 트래킹 및 평가를 위한 W&B Weave 툴킷을 의미하지 않습니다.

**속성:**
 
 - `entity` (str): 값이 기록된 프로젝트에 적절한 권한을 가진 entity입니다.
 - `project` (str): 값이 기록된 프로젝트입니다.
 - `table_name` (str): 테이블, DataFrame, 플롯 또는 값의 이름입니다.

---

## <kbd>class</kbd> `WeavePanel`
쿼리를 활용한 커스텀 콘텐츠를 표시할 수 있는 비어있는 쿼리 패널입니다.

API 이름의 "Weave"는 LLM 트래킹 및 평가를 위한 W&B Weave 툴킷을 의미하지 않습니다.

---

## <kbd>class</kbd> `WeavePanelArtifact`
W&B에 기록된 artifact를 보여주는 패널입니다.

API 이름의 "Weave"는 LLM 트래킹 및 평가를 위한 W&B Weave 툴킷을 의미하지 않습니다.

**속성:**
 
 - `artifact` (str): 가져올 artifact의 이름입니다.
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: artifact 패널에서 표시할 탭입니다.

---

## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`
W&B artifact에 기록된 버전 파일을 보여주는 패널입니다.

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 이름의 "Weave"는 LLM 트래킹 및 평가를 위한 W&B Weave 툴킷을 의미하지 않습니다.

**속성:**
 
 - `artifact` (str): 가져올 artifact의 이름입니다.
 - `version` (str): 가져올 artifact의 버전입니다.
 - `file` (str): 가져올 파일의 이름입니다.

---

## <kbd>class</kbd> `WeavePanelSummaryTable`
W&B Table, pandas DataFrame, 플롯 혹은 기타 값을 보여주는 패널입니다. 쿼리 형식은 다음과 같습니다.

```python
runs.summary['value']
```

API 이름의 "Weave"는 LLM 트래킹 및 평가를 위한 W&B Weave 툴킷을 의미하지 않습니다.

**속성:**
 
 - `table_name` (str): 테이블, DataFrame, 플롯 또는 값의 이름입니다.
