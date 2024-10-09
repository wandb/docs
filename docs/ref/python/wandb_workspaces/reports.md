import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Reports

<CTAButtons githubLink='https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py'/>

# <kbd>module</kbd> `wandb_workspaces.reports.v2`  
Weights & Biases Reports API와 프로그램적으로 작업하기 위한 Python 라이브러리입니다.

```python
# 임포트 방법
import wandb_workspaces.reports.v2
```

---

## <kbd>class</kbd> `BarPlot`  
2D 바 플롯을 보여주는 패널 오브젝트입니다.

**Attributes:**
 
- `title` (Optional[str]): 플롯 상단에 표시되는 텍스트.  
- `orientation Literal["v", "h"]`: 바 플롯의 방향. 세로("v") 또는 가로("h") 중 하나를 설정. 기본값은 가로("h").  
- `range_x` (Tuple[float | None, float | None]): x축 범위를 지정하는 튜플.  
- `title_x` (Optional[str]): x축의 라벨.  
- `title_y` (Optional[str]): y축의 라벨.  
- `groupby` (Optional[str]): 로그된 메트릭을 기반으로 러닝을 그룹화합니다. W&B 프로젝트에서 정보를 가져옵니다.  
- `groupby_aggfunc` (Optional[GroupAgg]): 지정된 함수로 러닝을 집계합니다. 옵션에는 "mean", "min", "max", "median", "sum", "samples", 또는 `None`이 포함됩니다.  
- `groupby_rangefunc` (Optional[GroupArea]): 범위를 기준으로 러닝을 그룹화합니다. 옵션에는 "minmax", "stddev", "stderr", "none", "samples", 또는 `None`이 포함됩니다.  
- `max_runs_to_show` (Optional[int]): 플롯에 표시할 최대 러닝 수.  
- `max_bars_to_show` (Optional[int]): 바 플롯에 표시할 최대 바의 수.  
- `custom_expressions` (Optional[LList[str]]): 바 플롯에 사용하는 사용자 지정 표현식 목록.  
- `legend_template` (Optional[str]): 범례의 템플릿.  
- `font_size` (Optional[FontSize]): 라인 플롯의 글꼴 크기. 옵션에는 "small", "medium", "large", "auto", 또는 `None`이 포함됩니다.  
- `line_titles` (Optional[dict]): 라인의 제목. 키는 라인 이름이고 값은 제목입니다.  
- `line_colors` (Optional[dict]): 라인의 색상. 키는 라인 이름이고 값은 색상입니다.

---

## <kbd>class</kbd> `BlockQuote`  
인용 블록입니다.

**Attributes:**
 
- `text` (str): 인용 블록의 텍스트.

---

## <kbd>class</kbd> `CalloutBlock`  
콜아웃 텍스트 블록입니다.

**Attributes:**
 
- `text` (str): 콜아웃 텍스트.

---

## <kbd>class</kbd> `CheckedList`  
체크박스가 있는 항목의 리스트입니다. `CheckedList` 내에 하나 이상의 `CheckedListItem`을 추가합니다.

**Attributes:**
 
- `items` (LList[CheckedListItem]): 하나 이상의 `CheckedListItem` 오브젝트의 리스트.

---

## <kbd>class</kbd> `CheckedListItem`  
체크박스가 있는 리스트 항목입니다. `CheckedList` 내에 하나 이상의 `CheckedListItem`을 추가합니다.

**Attributes:**
 
- `text` (str): 리스트 항목의 텍스트.  
- `checked` (bool): 체크박스가 선택되었는지 여부. 기본값은 `False`.

---

## <kbd>class</kbd> `CodeBlock`  
코드 블록입니다.

**Attributes:**
 
- `code` (str): 블록의 코드.  
- `language` (Optional[Language]): 코드의 언어. 지정된 언어가 구문 강조 표시에 사용됩니다. 기본값은 "python"입니다. 옵션에는 'javascript', 'python', 'css', 'json', 'html', 'markdown', 'yaml'이 포함됩니다.

---

## <kbd>class</kbd> `CodeComparer`  
두 개의 다른 run 사이의 코드를 비교하는 패널 오브젝트입니다.

**Attributes:**
 
- `diff` (Literal['split', 'unified']): 코드 차이를 표시하는 방법. 옵션에는 "split"과 "unified"가 포함됩니다.

---

## <kbd>class</kbd> `Config`  
run의 config 오브젝트에 로그된 메트릭입니다. Config 오브젝트는 일반적으로 `run.config[name] = ...`을 사용하거나 키-값 쌍의 사전으로 config를 전달하여 로그됩니다.

**Attributes:**
 
- `name` (str): 메트릭의 이름.

---

## <kbd>class</kbd> `CustomChart`  
Weave 쿼리로 정의된 사용자 지정 차트를 보여주는 패널입니다.

**Attributes:**
 
- `query` (dict): 사용자 지정 차트를 정의하는 쿼리. 키는 필드 이름이고 값은 쿼리입니다.  
- `chart_name` (str): 사용자 지정 차트의 제목.  
- `chart_fields` (dict): 플롯의 축을 정의하는 키-값 쌍. 키는 라벨이고 값은 메트릭입니다.  
- `chart_strings` (dict): 차트에서 문자열을 정의하는 키-값 쌍입니다.

---

### <kbd>classmethod</kbd> `from_table`

```python
from_table(
    table_name: str,
    chart_fields: dict = None,
    chart_strings: dict = None
)
```

테이블에서 사용자 정의 차트를 생성합니다.

**Arguments:**
 
- `table_name` (str): 테이블의 이름.  
- `chart_fields` (dict): 차트에서 표시할 필드.  
- `chart_strings` (dict): 차트에서 표시할 문자열.

---

## <kbd>class</kbd> `Gallery`  
Reports와 URL의 갤러리를 렌더링하는 블록입니다.

**Attributes:**
 
- `items` (List[Union[`GalleryReport`, `GalleryURL`]]): `GalleryReport`와 `GalleryURL` 오브젝트의 리스트.

---

## <kbd>class</kbd> `GalleryReport`  
갤러리에 있는 리포트를 참조합니다.

**Attributes:**
 
- `report_id` (str): 리포트의 ID.

---

## <kbd>class</kbd> `GalleryURL`  
외부 리소스에 대한 URL입니다.

**Attributes:**
 
- `url` (str): 리소스의 URL.  
- `title` (Optional[str]): 리소스의 제목.  
- `description` (Optional[str]): 리소스에 대한 설명.  
- `image_url` (Optional[str]): 표시할 이미지의 URL.

---

## <kbd>class</kbd> `GradientPoint`  
그레이디언트의 한 점입니다.

**Attributes:**
 
- `color`: 점의 색상.  
- `offset`: 그레이디언트에서 점의 위치. 값은 0과 100 사이여야 합니다.

---

## <kbd>class</kbd> `H1`  
지정된 텍스트가 있는 H1 헤딩입니다.

**Attributes:**
 
- `text` (str): 헤딩의 텍스트.  
- `collapsed_blocks` (Optional[LList["BlockTypes"]]): 헤딩이 축소될 때 표시할 블록.

---

## <kbd>class</kbd> `H2`  
지정된 텍스트가 있는 H2 헤딩입니다.

**Attributes:**
 
- `text` (str): 헤딩의 텍스트.  
- `collapsed_blocks` (Optional[LList["BlockTypes"]]): 헤딩이 축소될 때 표시할 하나 이상의 블록.

---

## <kbd>class</kbd> `H3`  
지정된 텍스트가 있는 H3 헤딩입니다.

**Attributes:**
 
- `text` (str): 헤딩의 텍스트.  
- `collapsed_blocks` (Optional[LList["BlockTypes"]]): 헤딩이 축소될 때 표시할 하나 이상의 블록.

---

## <kbd>class</kbd> `Heading`

---

## <kbd>class</kbd> `HorizontalRule`  
HTML 수평선입니다.

---

## <kbd>class</kbd> `Image`  
이미지를 렌더링하는 블록입니다.

**Attributes:**
 
- `url` (str): 이미지의 URL.  
- `caption` (str): 이미지의 캡션. 캡션은 이미지 아래에 나타납니다.

---

## <kbd>class</kbd> `InlineCode`  
인라인 코드입니다. 코드 이후에 줄 바꿈 문자를 추가하지 않습니다.

**Attributes:**
 
- `text` (str): 리포트에 표시하고자 하는 코드.

---

## <kbd>class</kbd> `InlineLatex`  
인라인 LaTeX Markdown입니다. LaTeX Markdown 이후에 줄 바꿈 문자를 추가하지 않습니다.

**Attributes:**
 
- `text` (str): 리포트에 표시하고자 하는 LaTeX Markdown.

---

## <kbd>class</kbd> `LatexBlock`  
LaTeX 텍스트 블록입니다.

**Attributes:**
 
- `text` (str): LaTeX 텍스트.

---

## <kbd>class</kbd> `Layout`  
리포트의 패널 레이아웃입니다. 패널의 크기와 위치를 조정합니다.

**Attributes:**
 
- `x` (int): 패널의 x 위치.  
- `y` (int): 패널의 y 위치.  
- `w` (int): 패널의 폭.  
- `h` (int): 패널의 높이.

---

## <kbd>class</kbd> `LinePlot`  
2D 라인 플롯을 가진 패널 오브젝트입니다.

**Attributes:**
 
- `title` (Optional[str]): 플롯 상단에 표시되는 텍스트.  
- `x` (Optional[MetricType]): W&B 프로젝트에 로그된 메트릭의 이름으로, 이 리포트는 이 정보에서 정보를 가져옵니다. 지정된 메트릭은 x축에 사용됩니다.  
- `y` (LList[MetricType]): W&B 프로젝트에 로그된 하나 이상의 메트릭으로, 이 리포트는 이 정보에서 정보를 가져옵니다. 지정된 메트릭은 y축에 사용됩니다.  
- `range_x` (Tuple[float | `None`, float | `None`]): x축의 범위를 지정하는 튜플.  
- `range_y` (Tuple[float | `None`, float | `None`]): y축의 범위를 지정하는 튜플.  
- `log_x` (Optional[bool]): x좌표를 10진 로그 스케일로 플롯합니다.  
- `log_y` (Optional[bool]): y좌표를 10진 로그 스케일로 플롯합니다.  
- `title_x` (Optional[str]): x축의 라벨.  
- `title_y` (Optional[str]): y축의 라벨.  
- `ignore_outliers` (Optional[bool]): `True`로 설정하면 이상값을 플롯하지 않습니다.  
- `groupby` (Optional[str]): W&B 프로젝트에 로그된 메트릭을 기반으로 러닝을 그룹화합니다. 이 리포트는 이 정보에서 정보를 가져옵니다.  
- `groupby_aggfunc` (Optional[GroupAgg]): 지정된 함수로 러닝을 집계합니다. 옵션에는 "mean", "min", "max", "median", "sum", "samples", 또는 `None`이 포함됩니다.  
- `groupby_rangefunc` (Optional[GroupArea]): 범위를 기준으로 러닝을 그룹화합니다. 옵션에는 "minmax", "stddev", "stderr", "none", "samples", 또는 `None`이 포함됩니다.  
- `smoothing_factor` (Optional[float]): 부드럽게 하기 위한 요인. 수용 값은 0과 1 사이입니다.  
- `smoothing_type Optional[SmoothingType]`: 지정된 분포를 기반으로 필터를 적용합니다. 옵션에는 "exponentialTimeWeighted", "exponential", "gaussian", "average", 또는 "none"이 포함됩니다.  
- `smoothing_show_original` (Optional[bool]): `True`로 설정하면 원본 데이터를 표시합니다.  
- `max_runs_to_show` (Optional[int]): 라인 플롯에 표시할 최대 러닝 수.  
- `custom_expressions` (Optional[LList[str]]): 데이터에 적용할 사용자 지정 표현식.  
- `plot_type Optional[LinePlotStyle]`: 생성할 라인 플롯의 유형. 옵션에는 "line", "stacked-area", 또는 "pct-area"가 포함됩니다.  
- `font_size Optional[FontSize]`: 라인 플롯의 글꼴 크기. 옵션에는 "small", "medium", "large", "auto", 또는 `None`이 포함됩니다.  
- `legend_position Optional[LegendPosition]`: 범례를 배치할 위치. 옵션에는 "north", "south", "east", "west", 또는 `None`이 포함됩니다.  
- `legend_template` (Optional[str]): 범례의 템플릿.  
- `aggregate` (Optional[bool]): `True`로 설정하면 데이터를 집계합니다.  
- `xaxis_expression` (Optional[str]): x축의 표현식.  
- `legend_fields` (Optional[LList[str]]): 범례에 포함할 필드.

---

## <kbd>class</kbd> `Link`  
URL에 대한 링크입니다.

**Attributes:**
 
- `text` (Union[str, TextWithInlineComments]): 링크의 텍스트.  
- `url` (str): 링크가 가리키는 URL.

---

## <kbd>class</kbd> `MarkdownBlock`  
Markdown 텍스트의 블록입니다. 일반적인 markdown 문법을 사용하는 텍스트를 작성하려면 유용합니다.

**Attributes:**
 
- `text` (str): Markdown 텍스트.

---

## <kbd>class</kbd> `MarkdownPanel`  
Markdown을 렌더링하는 패널입니다.

**Attributes:**
 
- `markdown` (str): Markdown 패널에 표시하고자 하는 텍스트.

---

## <kbd>class</kbd> `MediaBrowser`  
미디어 파일을 그리드 레이아웃으로 표시하는 패널입니다.

**Attributes:**
 
- `num_columns` (Optional[int]): 그리드의 열 수.  
- `media_keys` (LList[str]): 미디어 파일에 해당하는 미디어 키 목록.

---

## <kbd>class</kbd> `Metric`  
프로젝트에 로그된 리포트에서 표시할 메트릭입니다.

**Attributes:**
 
- `name` (str): 메트릭의 이름.

---

## <kbd>class</kbd> `OrderBy`  
정렬할 메트릭입니다.

**Attributes:**
 
- `name` (str): 메트릭의 이름.  
- `ascending` (bool): 오름차순으로 정렬할지 여부. 기본값은 `False`.

---

## <kbd>class</kbd> `OrderedList`  
번호 매긴 목록의 항목 리스트입니다.

**Attributes:**
 
- `items` (LList[str]): 하나 이상의 `OrderedListItem` 오브젝트의 리스트.

---

## <kbd>class</kbd> `OrderedListItem`  
순서 있는 리스트의 항목입니다.

**Attributes:**
 
- `text` (str): 리스트 항목의 텍스트.

---

## <kbd>class</kbd> `P`  
텍스트의 문단입니다.

**Attributes:**
 
- `text` (str): 문단의 텍스트.

---

## <kbd>class</kbd> `Panel`  
패널 그리드에서 시각화를 표시하는 패널입니다.

**Attributes:**
 
- `layout` (Layout): `Layout` 오브젝트.

---

## <kbd>class</kbd> `PanelGrid`  
Runset과 패널로 구성된 그리드입니다. `Runset`과 `Panel` 오브젝트를 추가하여 구성합니다.

사용 가능한 패널에는 `LinePlot`, `ScatterPlot`, `BarPlot`, `ScalarChart`, `CodeComparer`, `ParallelCoordinatesPlot`, `ParameterImportancePlot`, `RunComparer`, `MediaBrowser`, `MarkdownPanel`, `CustomChart`, `WeavePanel`, `WeavePanelSummaryTable`, `WeavePanelArtifactVersionedFile` 등이 포함됩니다.

**Attributes:**
 
- `runsets` (LList["Runset"]): 하나 이상의 `Runset` 오브젝트의 리스트.  
- `panels` (LList["PanelTypes"]): 하나 이상의 `Panel` 오브젝트의 리스트.  
- `active_runset` (int): 런셋 내에서 표시할 run 수. 기본값은 0으로 설정됩니다.  
- `custom_run_colors` (dict): 키가 run 이름이고 값이 16진수 값으로 지정된 색상인 키-값 쌍.

---

## <kbd>class</kbd> `ParallelCoordinatesPlot`  
평행 좌표 플롯을 보여주는 패널 오브젝트입니다.

**Attributes:**
 
- `columns` (LList[ParallelCoordinatesPlotColumn]): 하나 이상의 `ParallelCoordinatesPlotColumn` 오브젝트 리스트.  
- `title` (Optional[str]): 플롯 상단에 표시되는 텍스트.  
- `gradient` (Optional[LList[GradientPoint]]): 그레이디언트 포인트 리스트.  
- `font_size` (Optional[FontSize]): 라인 플롯의 글꼴 크기. 옵션에는 "small", "medium", "large", "auto", 또는 `None`이 포함됩니다.

---

## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`  
평행 좌표 플롯 내의 열입니다. 지정된 `metric`의 순서는 평행 좌표 플롯의 평행 축(x축)순서에 따라 결정됩니다.

**Attributes:**
 
- `metric` (str | Config | SummaryMetric): 리포트의 정보에서 정보를 가져오는 W&B 프로젝트에 로그된 메트릭의 이름.  
- `display_name` (Optional[str]): 메트릭의 이름.  
- `inverted` (Optional[bool]): 메트릭을 반전할지 여부.  
- `log` (Optional[bool]): 메트릭에 로그 변환을 적용할지 여부.

---

## <kbd>class</kbd> `ParameterImportancePlot`  
선택한 메트릭을 예측하는 데 있어 각 하이퍼파라미터가 얼마나 중요한지를 보여주는 패널입니다.

**Attributes:**
 
- `with_respect_to` (str): 파라미터 중요도를 비교하려는 메트릭. 일반적인 메트릭에는 손실, 정확도 등이 포함될 수 있습니다. 지정한 메트릭은 리포트가 정보를 가져오는 프로젝트 내에서 로그되어야 합니다.

---

## <kbd>class</kbd> `Report`  
Weights & Biases Report를 나타내는 오브젝트입니다. 반환된 오브젝트의 `blocks` 속성을 사용하여 리포트를 커스터마이징합니다. Report 오브젝트는 자동으로 저장되지 않습니다. `save()` 메소드를 사용하여 변경 사항을 지속시킵니다.

**Attributes:**
 
- `project` (str): 로드하려는 W&B 프로젝트의 이름입니다. 지정한 프로젝트는 리포트의 URL에 나타납니다.  
- `entity` (str): 리포트를 소유한 W&B 엔티티입니다. 엔티티는 리포트의 URL에 나타납니다.  
- `title` (str): 리포트의 제목입니다. 제목은 리포트 상단의 H1 헤딩으로 나타납니다.  
- `description` (str): 리포트에 대한 설명입니다. 설명은 리포트 제목 아래에 나타납니다.  
- `blocks` (LList[BlockTypes]): 하나 이상의 HTML 태그, 플롯, 그리드, runset 등을 포함하는 리스트입니다.  
- `width` (Literal['readable', 'fixed', 'fluid']): 리포트의 너비입니다. 옵션에는 'readable', 'fixed', 'fluid'가 포함됩니다.


---

#### <kbd>property</kbd> url

리포트가 호스팅된 URL입니다. 리포트 URL은 `https://wandb.ai/{entity}/{project_name}/reports/` 형태로 구성됩니다. 여기서 `{entity}`와 `{project_name}`은 각각 리포트가 속한 엔티티와 프로젝트의 이름을 나타냅니다.

---

### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str, as_model: bool = False)
```

현재 환경에 리포트를 로드합니다. 리포트가 호스팅된 URL을 전달합니다.

**Arguments:**

- `url` (str): 리포트가 호스팅된 URL입니다.  
- `as_model` (bool): True로 설정하면 Report 객체 대신 모델 객체를 반환합니다. 기본값은 `False`입니다.

---

### <kbd>method</kbd> `save`

```python
save(draft: bool = False, clone: bool = False)
```

리포트 오브젝트에서 수행한 변경 사항을 지속시킵니다.

---

### <kbd>method</kbd> `to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

해당 리포트를 표시하는 iframe을 포함한 HTML을 생성합니다. 일반적으로 Python 노트북 내에서 사용됩니다.

**Arguments:**
 
- `height` (int): iframe의 높이.  
- `hidden` (bool): True로 설정하면 iframe을 숨깁니다. 기본값은 `False`.

---

## <kbd>class</kbd> `RunComparer`  
리포트가 정보를 가져오는 프로젝트의 다양한 run에서 메트릭을 비교하는 패널입니다.

**Attributes:**

- `diff_only` (Optional[Literal["split", `True`]]): 프로젝트의 run 간 차이만 표시합니다. W&B Report UI에서 이 기능을 켜고 끌 수 있습니다.

---

## <kbd>class</kbd> `Runset`  
패널 그리드에 표시할 run의 집합입니다.

**Attributes:**
 
- `entity` (str): 프로젝트에서 run을 소유하거나 적절한 권한을 가진 엔티티입니다.  
- `project` (str): 러닝이 저장된 프로젝트의 이름입니다.  
- `name` (str): Runset의 이름입니다. 기본값으로 `Run set`으로 설정됩니다.  
- `query` (str): run을 필터링하기 위한 쿼리 문자열입니다.  
- `filters` (Optional[str]): run을 필터링하기 위한 필터 문자열입니다.  
- `groupby` (LList[str]): 그룹화할 메트릭 이름들의 리스트입니다.  
- `order` (LList[OrderBy]): 정렬할 `OrderBy` 오브젝트들의 리스트입니다.  
- `custom_run_colors` (LList[OrderBy]): run ID를 색상에 매핑하는 사전입니다.

---

## <kbd>class</kbd> `RunsetGroup`  
runset의 그룹을 보여주는 UI 요소입니다.

**Attributes:**
 
- `runset_name` (str): runset의 이름입니다.   
- `keys` (Tuple[RunsetGroupKey, ...]): 그룹화할 키들입니다. 한 개 이상의 `RunsetGroupKey` 오브젝트를 전달하여 그룹화합니다.

---

## <kbd>class</kbd> `RunsetGroupKey`  
metric 타입과 값으로 runset을 그룹화합니다. `RunsetGroup`의 일부입니다. 그룹화할 metric 타입과 값을 키-값 쌍으로 지정합니다.

**Attributes:**
 
- `key` (Type[str] | Type[Config] | Type[SummaryMetric] | Type[Metric]): 그룹화할 metric 타입입니다.   
- `value` (str): 그룹화할 metric의 값입니다.

---

## <kbd>class</kbd> `ScalarChart`  
스칼라 차트를 보여주는 패널 오브젝트입니다.

**Attributes:**
 
- `title` (Optional[str]): 플롯 상단에 표시되는 텍스트입니다.   
- `metric` (MetricType): 리포트가 정보를 가져오는 W&B 프로젝트에 로그된 메트릭의 이름입니다.   
- `groupby_aggfunc` (Optional[GroupAgg]): 지정된 함수로 run을 집계합니다. 옵션에는 "mean", "min", "max", "median", "sum", "samples", 또는 `None`이 포함됩니다.   
- `groupby_rangefunc` (Optional[GroupArea]): 범위에 따라 run을 그룹화합니다. 옵션에는 "minmax", "stddev", "stderr", "none", "samples", 또는 `None`이 포함됩니다.   
- `custom_expressions` (Optional[LList[str]]): 스칼라 차트에서 사용할 사용자 지정 표현식의 리스트입니다.    
- `legend_template` (Optional[str]): 범례의 템플릿입니다.   
- `font_size Optional[FontSize]`: 라인 플롯의 글꼴 크기입니다. 옵션에는 "small", "medium", "large", "auto", 또는 `None`이 포함됩니다. 

---

## <kbd>class</kbd> `ScatterPlot`  
2D 또는 3D 산점도를 보여주는 패널 오브젝트입니다.

**Arguments:**

- `title` (Optional[str]): 플롯 상단에 표시되는 텍스트입니다.  
- `x Optional[SummaryOrConfigOnlyMetric]`: 리포트가 정보를 가져오는 W&B 프로젝트에 로그된 메트릭의 이름입니다. 지정된 메트릭은 x축에 사용됩니다.  
- `y Optional[SummaryOrConfigOnlyMetric]`: 리포트가 정보를 가져오는 W&B 프로젝트에 로그된 하나 이상의 메트릭입니다. 지정된 메트릭은 y축에 플롯됩니다.  
- `range_x` (Tuple[float | `None`, float | `None`]): x축의 범위를 지정하는 튜플입니다.  
- `range_y` (Tuple[float | `None`, float | `None`]): y축의 범위를 지정하는 튜플입니다.  
- `range_z` (Tuple[float | `None`, float | `None`]): z축의 범위를 지정하는 튜플입니다.  
- `log_x` (Optional[bool]): x좌표를 10진 로그 스케일로 플롯합니다.  
- `log_y` (Optional[bool]): y좌표를 10진 로그 스케일로 플롯합니다.  
- `log_z` (Optional[bool]): z좌표를 10진 로그 스케일로 플롯합니다.  
- `running_ymin` (Optional[bool]): 이동 평균 또는 롤링 평균을 적용합니다.  
- `running_ymax` (Optional[bool]): 이동 평균 또는 롤링 평균을 적용합니다.  
- `running_ymean` (Optional[bool]): 이동 평균 또는 롤링 평균을 적용합니다.  
- `legend_template` (Optional[str]): 범례의 형식을 지정하는 문자열입니다.    
- `gradient` (Optional[LList[GradientPoint]]): 플롯의 색상 그라데이션을 지정하는 그라데이션 포인트의 리스트입니다.  
- `font_size` (Optional[FontSize]): 라인 플롯의 글꼴 크기입니다. 옵션에는 "small", "medium", "large", "auto", 또는 `None`이 포함됩니다.  
- `regression` (Optional[bool]): `True`로 설정하면 산점도에 회귀선을 플롯합니다. 

---

## <kbd>class</kbd> `SoundCloud`  
SoundCloud 플레이어를 렌더링하는 블록입니다.

**Attributes:**
 
- `html` (str): SoundCloud 플레이어를 임베드하기 위한 HTML 코드입니다.

---

## <kbd>class</kbd> `Spotify`  
Spotify 플레이어를 렌더링하는 블록입니다.

**Attributes:**
 
- `spotify_id` (str): 트랙 또는 플레이리스트의 Spotify ID입니다.

---

## <kbd>class</kbd> `SummaryMetric`  
리포트에서 표시할 요약 메트릭입니다.

**Attributes:**
 
- `name` (str): 메트릭의 이름입니다.

---

## <kbd>class</kbd> `TableOfContents`  
리포트 내의 H1, H2, 및 H3 HTML 블록을 사용하여 섹션과 하위 섹션의 리스트를 포함하는 블록입니다.

---

## <kbd>class</kbd> `TextWithInlineComments`  
인라인 코멘트가 있는 텍스트 블록입니다.

**Attributes:**
 
- `text` (str): 블록의 텍스트입니다.

---

## <kbd>class</kbd> `Twitter`  
트위터 피드를 표시하는 블록입니다.

**Attributes:**
 
- `html` (str): 트위터 피드를 표시하는 HTML 코드입니다.

---

## <kbd>class</kbd> `UnorderedList`  
불릿 목록의 항목 리스트입니다.

**Attributes:**
 
- `items` (LList[str]): 하나 이상의 `UnorderedListItem` 오브젝트의 리스트입니다.

---

## <kbd>class</kbd> `UnorderedListItem`  
순서 없는 리스트의 항목입니다.

**Attributes:**
 
- `text` (str): 리스트 항목의 텍스트입니다.

---

## <kbd>class</kbd> `Video`  
비디오를 렌더링하는 블록입니다.

**Attributes:**
 
- `url` (str): 비디오의 URL입니다.

---

## <kbd>class</kbd> `WeaveBlockArtifact`  
W&B에 로그된 아티팩트를 보여주는 블록입니다. 쿼리는 다음과 같은 형태를 가집니다:

```python
project('entity', 'project').artifact('artifact-name')
```

API 이름의 "Weave"는 LLM을 추적하고 평가하는 데 사용되는 W&B Weave 도구킷을 나타내지 않습니다.

**Attributes:**

- `entity` (str): 아티팩트를 저장하는 프로젝트에 적절한 권한을 가진 소유자 또는 엔티티입니다.  
- `project` (str): 아티팩트를 저장하는 프로젝트입니다.  
- `artifact` (str): 조회하려는 아티팩트의 이름입니다.  
- `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: 아티팩트 패널에 표시할 탭입니다.

---

## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`  
W&B 아티팩트에 로그된 버전 파일을 보여주는 블록입니다. 쿼리는 다음과 같은 형태를 가집니다:

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 이름의 "Weave"는 LLM을 추적하고 평가하는 데 사용되는 W&B Weave 도구킷을 나타내지 않습니다.

**Attributes:**
 
- `entity` (str): 아티팩트를 저장하는 프로젝트에 적절한 권한을 가진 소유자 또는 엔티티입니다.  
- `project` (str): 아티팩트를 저장하는 프로젝트입니다.  
- `artifact` (str): 조회하려는 아티팩트의 이름입니다.  
- `version` (str): 조회하려는 아티팩트의 버전입니다.  
- `file` (str): 아티팩트 내에 저장된 조회하려는 파일의 이름.

---

## <kbd>class</kbd> `WeaveBlockSummaryTable`  
W&B에 로그된 W&B Table, pandas DataFrame, 플롯 또는 다른 값을 보여주는 블록입니다. 쿼리는 다음과 같은 형태를 가집니다:

```python
project('entity', 'project').runs.summary['value']
```

API 이름의 "Weave"는 LLM을 추적하고 평가하는 데 사용되는 W&B Weave 도구킷을 나타내지 않습니다.

**Attributes:**

- `entity` (str): 값이 로그된 프로젝트에 적절한 권한을 가진 소유자 또는 엔티티입니다.  
- `project` (str): 값이 로그된 프로젝트입니다.  
- `table_name` (str): Table, DataFrame, 플롯 또는 값의 이름입니다.

---

## <kbd>class</kbd> `WeavePanel`  
쿼리를 사용하여 사용자 지정 콘텐츠를 표시할 수 있는 빈 쿼리 패널입니다.

API 이름의 "Weave"는 LLM을 추적하고 평가하는 데 사용되는 W&B Weave 도구킷을 나타내지 않습니다.

---

## <kbd>class</kbd> `WeavePanelArtifact`  
W&B에 로그된 아티팩트를 보여주는 패널입니다.

API 이름의 "Weave"는 LLM을 추적하고 평가하는 데 사용되는 W&B Weave 도구킷을 나타내지 않습니다.

**Attributes:**

- `artifact` (str): 조회하려는 아티팩트의 이름입니다.  
- `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: 아티팩트 패널에 표시할 탭입니다.

---

## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`  
W&B 아티팩트에 로그된 버전 파일을 보여주는 패널입니다.

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
```

API 이름의 "Weave"는 LLM을 추적하고 평가하는 데 사용되는 W&B Weave 도구킷을 나타내지 않습니다.

**Attributes:**

- `artifact` (str): 조회하려는 아티팩트의 이름입니다.  
- `version` (str): 조회하려는 아티팩트의 버전입니다.  
- `file` (str): 아티팩트 내에 저장된 조회하려는 파일의 이름입니다.

---

## <kbd>class</kbd> `WeavePanelSummaryTable`  
W&B에 로그된 W&B Table, pandas DataFrame, 플롯 또는 다른 값을 보여주는 패널입니다. 쿼리는 다음과 같은 형태를 가집니다:

```python
runs.summary['value']
```

API 이름의 "Weave"는 LLM을 추적하고 평가하는 데 사용되는 W&B Weave 도구킷을 나타내지 않습니다.

**Attributes:**

- `table_name` (str): Table, DataFrame, 플롯 또는 값의 이름입니다.
