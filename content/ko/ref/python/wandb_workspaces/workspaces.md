---
title: 워크스페이스
menu:
  reference:
    identifier: ko-ref-python-wandb_workspaces-workspaces
---

{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/workspaces/interface.py" >}}




{{% alert %}}
W&B Report 및 Workspace API는 Public Preview 단계입니다.
{{% /alert %}}


# <kbd>module</kbd> `wandb_workspaces.workspaces`
W&B Workspace API를 프로그래밍 방식으로 다루기 위한 Python 라이브러리입니다.

```python
# 임포트 방법
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
런셋(왼쪽 바)에서 개별 run에 대한 설정 클래스입니다.



**속성:**
 
 - `color` (str): UI에서 run에 표시되는 색상입니다. hex 색상(#ff0000), css 색상(red), 또는 rgb(rgb(255, 0, 0)) 형식 사용 가능
 - `disabled` (bool): run이 비활성화(아이콘이 닫힌 눈) 상태인지 여부입니다. 기본값은 `False`입니다.







---



## <kbd>class</kbd> `RunsetSettings`
워크스페이스 내 runset(왼쪽 바의 runs 리스트)에 대한 설정 클래스입니다.



**속성:**
 
 - `query` (str): runset을 필터링하는 쿼리(정규표현식 가능, 아래 파라미터 참고)
 - `regex_query` (bool): 위 쿼리가 정규표현식인지 여부를 지정합니다. 기본값은 `False`입니다.
 - `filters` `(LList[expr.FilterExpr])`: runset에 적용할 필터의 리스트입니다. 필터들은 AND 조건으로 결합됩니다. 자세한 필터 생성 방법은 FilterExpr를 참조하세요.
 - `groupby` `(LList[expr.MetricType])`: runset에서 그룹핑할 메트릭의 리스트입니다. `Metric`, `Summary`, `Config`, `Tags`, `KeysInfo` 중에서 설정할 수 있습니다.
 - `order` `(LList[expr.Ordering])`: runset에 적용할 메트릭 및 정렬 기준의 리스트입니다.
 - `run_settings` `(Dict[str, RunSettings])`: run의 ID를 키로, RunSettings 오브젝트를 값으로 가지는 run 설정 딕셔너리입니다.







---



## <kbd>class</kbd> `Section`
워크스페이스 내에서 하나의 섹션을 나타냅니다.



**속성:**
 
 - `name` (str): 섹션의 이름/제목입니다.
 - `panels` `(LList[PanelTypes])`: 섹션 내 패널의 순서 있는 리스트입니다. 기본적으로 첫 번째가 좌측 상단, 마지막이 우측 하단입니다.
 - `is_open` (bool): 섹션이 열려 있는지 닫혀 있는지 여부입니다. 기본값은 닫힘입니다.
 - `layout_settings` `(Literal[`standard`, `custom`])`: 해당 섹션의 패널 레이아웃 설정입니다.
 - `panel_settings`: 해당 섹션 내 모든 패널에 적용되는 패널 레벨의 설정입니다. `Section`의 `WorkspaceSettings`와 유사합니다.







---



## <kbd>class</kbd> `SectionLayoutSettings`
섹션의 패널 레이아웃 설정을 담당합니다. 주로 W&B App 워크스페이스 UI의 섹션 우측 상단에서 볼 수 있습니다.



**속성:**
 
 - `layout` `(Literal[`standard`, `custom`])`: 섹션 내 패널 레이아웃 유형입니다. `standard`는 기본 그리드 레이아웃, `custom`은 개별 패널에 맞춤 레이아웃을 허용합니다.
 - `columns` (int): standard 레이아웃에서 컬럼(열)의 개수입니다. 기본값은 3입니다.
 - `rows` (int): standard 레이아웃에서 행의 개수입니다. 기본값은 2입니다.







---



## <kbd>class</kbd> `SectionPanelSettings`
섹션 단위의 패널 설정으로, 섹션의 `WorkspaceSettings`와 유사합니다.

이 설정은 더 세분화된 패널 설정에 의해 덮어써질 수 있습니다. 우선순위는 Section < Panel 입니다.



**속성:**
 
 - `x_axis` (str): X축 메트릭 이름 설정. 기본값은 `Step`입니다.
 - `x_min Optional[float]`: X축의 최소값입니다.
 - `x_max Optional[float]`: X축의 최대값입니다.
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']): 모든 패널에 적용할 스무딩 타입입니다.
 - `smoothing_weight` (int): 모든 패널에 적용할 스무딩 가중치입니다.







---



## <kbd>class</kbd> `Workspace`
W&B 워크스페이스를 나타내며, 섹션, 설정, runset 구성 등을 포함합니다.



**속성:**
 
 - `entity` (str): 이 워크스페이스를 저장할 entity(보통 사용자 혹은 팀 이름)입니다.
 - `project` (str): 이 워크스페이스를 저장할 project입니다.
 - `name`: 워크스페이스의 이름입니다.
 - `sections` `(LList[Section])`: 워크스페이스 내 섹션의 순서 있는 리스트입니다. 첫 번째 섹션이 워크스페이스 상단에 표시됩니다.
 - `settings` `(WorkspaceSettings)`: 워크스페이스 전체 설정으로, UI에서 워크스페이스 상단에 보이는 설정들입니다.
 - `runset_settings` `(RunsetSettings)`: 워크스페이스 내 runset(왼쪽 runs 바)에 대한 설정입니다.


---

#### <kbd>property</kbd> url

W&B 앱에서 이 워크스페이스의 URL입니다.



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



**반환값:**
 
 - `Workspace`: 내부적으로 저장된 이름과 ID가 포함된 갱신된 워크스페이스 오브젝트입니다.

---



### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

현재 워크스페이스를 새로운 뷰로 W&B에 저장합니다.



**반환값:**
 
 - `Workspace`: 내부적으로 저장된 이름과 ID가 포함된 갱신된 워크스페이스 오브젝트입니다.

---



## <kbd>class</kbd> `WorkspaceSettings`
워크스페이스 전체 설정으로, 보통 UI에서 워크스페이스 상단에 표시됩니다.

이 오브젝트는 x축 설정, 스무딩, 이상치, 패널, 툴팁, run, 패널 쿼리 바 설정을 포함합니다.

여기서 적용된 설정은 더 세분화된 Section 및 Panel 설정에 의해 아래의 우선순위로 덮어써질 수 있습니다: Workspace < Section < Panel



**속성:**
 
 - `x_axis` (str): X축 메트릭 이름 설정입니다.
 - `x_min` `(Optional[float])`: X축 최소값입니다.
 - `x_max` `(Optional[float])`: X축 최대값입니다.
 - `smoothing_type` `(Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none'])`: 모든 패널에 적용할 스무딩 타입입니다.
 - `smoothing_weight` (int): 모든 패널에 적용할 스무딩 가중치입니다.
 - `ignore_outliers` (bool): 모든 패널에서 이상치를 무시할지 여부입니다.
 - `sort_panels_alphabetically` (bool): 모든 섹션의 패널을 이름순으로 정렬할지 여부입니다.
 - `group_by_prefix` `(Literal[`first`, `last`])`: 패널을 prefix(접두사) 기준으로 그룹핑합니다. `first` 또는 `last` 지정 가능. 기본값은 `last`입니다.
 - `remove_legends_from_panels` (bool): 모든 패널에서 범례를 제거할지 여부입니다.
 - `tooltip_number_of_runs` `(Literal[`default`, `all`, `none`])`: 툴팁에 몇 개의 run을 표시할지 설정합니다.
 - `tooltip_color_run_names` (bool): 툴팁에서 run 이름 색상을 runset과 일치시켜 표시할지(True) 여부입니다. 기본값은 `True`입니다.
 - `max_runs` (int): 패널별로 표시할 수 있는 최대 run 수(첫 10개의 run)입니다.
 - `point_visualization_method` `(Literal[`line`, `point`, `line_point`])`: 포인트 시각화 방식입니다.
 - `panel_search_query` (str): 패널 검색 바의 쿼리(정규표현식 가능)입니다.
 - `auto_expand_panel_search_results` (bool): 패널 검색 결과를 자동으로 확장할지 여부입니다.
