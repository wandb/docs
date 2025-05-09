---
title: Workspaces
menu:
  reference:
    identifier: ko-ref-python-wandb_workspaces-workspaces
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/workspaces/interface.py" >}}






# <kbd>module</kbd> `wandb_workspaces.workspaces`
W&B Workspace API를 프로그래밍 방식으로 사용하기 위한 Python 라이브러리입니다.

```python
# 가져오는 방법
import wandb_workspaces.workspaces as ws

# 워크스페이스 생성 예시
ws.Workspace(
     name="Example W&B Workspace",
     entity="entity", # 워크스페이스를 소유한 entity
     project="project", # 워크스페이스가 연결된 project
     sections=[
         ws.Section(
             name="Validation Metrics",
             panels=[
                 wr.LinePlot(x="Step", y=["val_loss"]),
                 wr.BarPlot(metrics=["val_accuracy"]),
                 wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
             ],
             is_open=True,
         ),
     ],
)
workspace.save()
```

---



## <kbd>class</kbd> `RunSettings`
runset의 run에 대한 설정입니다 (왼쪽 막대).

**속성:**
 
 - `color` (str): UI에서 run의 색상입니다. 16진수 (#ff0000), CSS 색상 (red) 또는 rgb (rgb(255, 0, 0))가 될 수 있습니다.
 - `disabled` (bool): run이 비활성화되었는지 여부입니다 (UI에서 눈이 감김). 기본값은 `False`로 설정됩니다.

---



## <kbd>class</kbd> `RunsetSettings`
워크스페이스의 runset (run을 포함하는 왼쪽 막대)에 대한 설정입니다.

**속성:**
 
 - `query` (str): runset을 필터링하는 쿼리입니다 (정규식 표현일 수 있음, 다음 매개변수 참조).
 - `regex_query` (bool): 쿼리 (위)가 정규식 표현인지 여부를 제어합니다. 기본값은 `False`로 설정됩니다.
 - `filters` `(LList[expr.FilterExpr])`: runset에 적용할 필터 목록입니다. 필터는 AND로 결합됩니다. 필터 생성에 대한 자세한 내용은 FilterExpr를 참조하세요.
 - `groupby` `(LList[expr.MetricType])`: runset에서 그룹화할 메트릭 목록입니다. `Metric`, `Summary`, `Config`, `Tags` 또는 `KeysInfo`로 설정됩니다.
 - `order` `(LList[expr.Ordering])`: runset에 적용할 메트릭 및 순서 지정 목록입니다.
 - `run_settings` `(Dict[str, RunSettings])`: run 설정 사전입니다. 여기서 키는 run의 ID이고 값은 RunSettings 오브젝트입니다.

---



## <kbd>class</kbd> `Section`
워크스페이스의 섹션을 나타냅니다.

**속성:**
 
 - `name` (str): 섹션의 이름/제목입니다.
 - `panels` `(LList[PanelTypes])`: 섹션의 패널 순서가 지정된 목록입니다. 기본적으로 첫 번째는 왼쪽 상단이고 마지막은 오른쪽 하단입니다.
 - `is_open` (bool): 섹션이 열려 있는지 닫혀 있는지 여부입니다. 기본값은 닫힘입니다.
 - `layout_settings` `(Literal[`standard`, `custom`])`: 섹션의 패널 레이아웃에 대한 설정입니다.
 - `panel_settings`: 섹션의 모든 패널에 적용되는 패널 수준 설정입니다. `Section`에 대한 `WorkspaceSettings`와 유사합니다.

---



## <kbd>class</kbd> `SectionLayoutSettings`
섹션의 패널 레이아웃 설정입니다. 일반적으로 W&B App Workspace UI의 섹션 상단 오른쪽에 표시됩니다.

**속성:**
 
 - `layout` `(Literal[`standard`, `custom`])`: 섹션의 패널 레이아웃입니다. `standard`는 기본 그리드 레이아웃을 따르고, `custom`은 개별 패널 설정에 의해 제어되는 패널별 레이아웃을 허용합니다.
 - `columns` (int): 표준 레이아웃에서 레이아웃의 열 수입니다. 기본값은 3입니다.
 - `rows` (int): 표준 레이아웃에서 레이아웃의 행 수입니다. 기본값은 2입니다.

---



## <kbd>class</kbd> `SectionPanelSettings`
섹션에 대한 패널 설정입니다. 섹션에 대한 `WorkspaceSettings`와 유사합니다.

여기에 적용된 설정은 Section < Panel 우선 순위로 더 세분화된 패널 설정으로 재정의할 수 있습니다.

**속성:**
 
 - `x_axis` (str): X축 메트릭 이름 설정입니다. 기본적으로 `Step`으로 설정됩니다.
 - `x_min Optional[float]`: X축의 최소값입니다.
 - `x_max Optional[float]`: X축의 최대값입니다.
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']): 모든 패널에 적용되는 평활화 유형입니다.
 - `smoothing_weight` (int): 모든 패널에 적용되는 평활화 가중치입니다.

---



## <kbd>class</kbd> `Workspace`
섹션, 설정 및 run set 구성을 포함하는 W&B 워크스페이스를 나타냅니다.

**속성:**
 
 - `entity` (str): 이 워크스페이스가 저장될 entity입니다 (일반적으로 사용자 또는 팀 이름).
 - `project` (str): 이 워크스페이스가 저장될 project입니다.
 - `name`: 워크스페이스의 이름입니다.
 - `sections` `(LList[Section])`: 워크스페이스의 섹션 순서가 지정된 목록입니다. 첫 번째 섹션이 워크스페이스 상단에 있습니다.
 - `settings` `(WorkspaceSettings)`: 워크스페이스에 대한 설정입니다. 일반적으로 UI의 워크스페이스 상단에 표시됩니다.
 - `runset_settings` `(RunsetSettings)`: 워크스페이스의 runset (run을 포함하는 왼쪽 막대)에 대한 설정입니다.

---

#### <kbd>property</kbd> url

W&B 앱의 워크스페이스 URL입니다.

---

### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str)
```

URL에서 워크스페이스를 가져옵니다.

---

### <kbd>method</kbd> `save`

```python
save()
```

현재 워크스페이스를 W&B에 저장합니다.

**반환 값:**
 
 - `Workspace`: 저장된 내부 이름 및 ID로 업데이트된 워크스페이스입니다.

---

### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

현재 워크스페이스를 W&B에 새 뷰로 저장합니다.

**반환 값:**
 
 - `Workspace`: 저장된 내부 이름 및 ID로 업데이트된 워크스페이스입니다.

---

## <kbd>class</kbd> `WorkspaceSettings`
워크스페이스에 대한 설정입니다. 일반적으로 UI의 워크스페이스 상단에 표시됩니다.

이 오브젝트에는 X축, 평활화, 이상값, 패널, 툴팁, run 및 패널 쿼리 막대에 대한 설정이 포함됩니다.

여기에 적용된 설정은 Workspace < Section < Panel 우선 순위로 더 세분화된 섹션 및 패널 설정으로 재정의할 수 있습니다.

**속성:**
 
 - `x_axis` (str): X축 메트릭 이름 설정입니다.
 - `x_min` `(Optional[float])`: X축의 최소값입니다.
 - `x_max` `(Optional[float])`: X축의 최대값입니다.
 - `smoothing_type` `(Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none'])`: 모든 패널에 적용되는 평활화 유형입니다.
 - `smoothing_weight` (int): 모든 패널에 적용되는 평활화 가중치입니다.
 - `ignore_outliers` (bool): 모든 패널에서 이상값을 무시합니다.
 - `sort_panels_alphabetically` (bool): 모든 섹션에서 패널을 알파벳순으로 정렬합니다.
 - `group_by_prefix` `(Literal[`first`, `last`])`: 패널을 처음 또는 마지막 접두사 (처음 또는 마지막)로 그룹화합니다. 기본값은 `last`로 설정됩니다.
 - `remove_legends_from_panels` (bool): 모든 패널에서 범례를 제거합니다.
 - `tooltip_number_of_runs` `(Literal[`default`, `all`, `none`])`: 툴팁에 표시할 run 수입니다.
 - `tooltip_color_run_names` (bool): 툴팁에서 run 이름을 runset (True)과 일치하도록 색상을 지정할지 여부 (False)입니다. 기본값은 `True`로 설정됩니다.
 - `max_runs` (int): 패널당 표시할 최대 run 수입니다 (이는 runset의 처음 10개 run이 됩니다).
 - `point_visualization_method` `(Literal[`line`, `point`, `line_point`])`: 포인트 시각화 방법입니다.
 - `panel_search_query` (str): 패널 검색 막대에 대한 쿼리입니다 (정규식 표현일 수 있음).
 - `auto_expand_panel_search_results` (bool): 패널 검색 결과를 자동으로 확장할지 여부입니다.
