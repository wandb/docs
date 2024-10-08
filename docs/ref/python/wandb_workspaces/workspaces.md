import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Workspaces

<CTAButtons githubLink='https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/workspaces/interface.py'/>

# <kbd>module</kbd> `wandb_workspaces.workspaces`
W&B Workspace API와 프로그램적으로 작업하기 위한 파이썬 라이브러리입니다.

```python
# How to import
import wandb_workspaces.workspaces
```

---

## <kbd>class</kbd> `RunSettings`
runset(왼쪽 바) 안의 run에 대한 설정입니다.

**속성:**
 
 - `color`: UI에서 run의 색상입니다. 16진수(#ff0000), CSS 색상(red), 또는 RGB(rgb(255, 0, 0))로 지정할 수 있습니다.
 - `disabled`: run이 비활성화되었는지 여부입니다(UI에서 눈이 감긴 상태).

---

## <kbd>class</kbd> `RunsetSettings`
워크스페이스 내 runset(왼쪽 바에 있는 runs)의 설정입니다.

**속성:**
 
 - `query`: runset을 필터링하기 위한 쿼리입니다(정규 표현식을 사용할 수 있습니다. 다음 매개 변수 참조).
 - `regex_query`: 쿼리(위)가 정규 표현식인지 여부를 제어합니다.
 - `filters`: runset에 적용할 필터 목록입니다. 필터는 AND로 결합됩니다. 필터를 생성하는 방법에 대한 자세한 정보는 FilterExpr를 참조하세요.
 - `groupby`: runset에서 그룹으로 묶을 메트릭 목록입니다.
 - `order`: runset에 적용할 메트릭 및 정렬 순서 목록입니다.
 - `run_settings`: run 설정의 사전이며, 키는 run의 ID이고 값은 RunSettings 오브젝트입니다.

---

## <kbd>class</kbd> `Section`
워크스페이스의 섹션을 나타냅니다.

**속성:**
 
 - `name`: 섹션의 이름/제목입니다.
 - `panels`: 섹션의 패널 순서 목록입니다. 기본적으로 첫 번째는 좌상단, 마지막은 우하단입니다.
 - `is_open`: 섹션이 열려 있는지 닫혀 있는지 여부입니다. 기본값은 닫힌 상태입니다.
 - `layout_settings`: 섹션 내 패널 레이아웃에 대한 설정입니다.
 - `panel_settings`: 이 섹션의 모든 패널에 적용되는 패널 수준의 설정으로, WorkspaceSettings와 유사합니다.

---

## <kbd>class</kbd> `SectionLayoutSettings`
섹션의 패널 레이아웃 설정으로, 일반적으로 W&B 앱 워크스페이스 UI의 섹션 오른쪽 상단에서 볼 수 있습니다.

**속성:**
 
 - `layout`: 표준 레이아웃에서 레이아웃의 열 수입니다.
 - `columns`: 표준 레이아웃에서 레이아웃의 열 수입니다.
 - `rows`: 표준 레이아웃에서 레이아웃의 행 수입니다.

---

## <kbd>class</kbd> `SectionPanelSettings`
섹션에 대한 패널 설정으로, 이 섹션에 대한 `WorkspaceSettings`과 유사합니다.

여기에 적용된 설정은 더 세분화된 패널 설정에 의해 우선순위에 따라 덮어쓰일 수 있습니다: Section < Panel.

**속성:**
 
 - `x_axis`: X축 메트릭 이름 설정입니다.
 - `x_min`: x축의 최소 값입니다.
 - `x_max`: x축의 최대 값입니다.
 - `smoothing_type`: 모든 패널에 적용되는 스무딩 유형입니다.
 - `smoothing_weight`: 모든 패널에 적용되는 스무딩 가중치입니다.

---

## <kbd>class</kbd> `Workspace`
섹션, 설정 및 run 세트 구성을 포함하는 W&B 워크스페이스를 나타냅니다.

**속성:**
 
 - `entity`: 이 워크스페이스가 저장될 엔티티(일반적으로 사용자 또는 팀 이름)입니다.
 - `project`: 이 워크스페이스가 저장될 프로젝트입니다.
 - `name`: 워크스페이스의 이름입니다.
 - `sections`: 워크스페이스 내 섹션의 순서 목록입니다. 첫 번째 섹션은 워크스페이스 상단에 있습니다.
 - `settings`: UI에서 일반적으로 워크스페이스 상단에 표시되는 워크스페이스 설정입니다.
 - `runset_settings`: 워크스페이스 내 runset(왼쪽 바에 포함된 runs)에 대한 설정입니다.

---

#### <kbd>property</kbd> url

W&B 앱에서의 워크스페이스 URL입니다.

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
 
 - `Workspace`: 저장된 내부 이름과 ID를 갖는 업데이트된 워크스페이스입니다.

---

### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

현재 워크스페이스를 W&B에 새로운 보기로 저장합니다.

**반환값:**
 
 - `Workspace`: 저장된 내부 이름과 ID를 갖는 업데이트된 워크스페이스입니다.

---

## <kbd>class</kbd> `WorkspaceSettings`
워크스페이스에 대한 설정으로, 일반적으로 UI에서 워크스페이스 상단에 표시됩니다.

이 오브젝트에는 x축, 스무딩, 이상치, 패널, 툴팁, runs, 패널 쿼리 바에 대한 설정이 포함됩니다.

여기에 적용된 설정은 더 세분화된 섹션 및 패널 설정에 의해 다음 우선순위에 따라 덮어쓰일 수 있습니다: Workspace < Section < Panel.

**속성:**
 
 - `x_axis`: X축 메트릭 이름 설정입니다.
 - `x_min`: x축의 최소 값입니다.
 - `x_max`: x축의 최대 값입니다.
 - `smoothing_type`: 모든 패널에 적용되는 스무딩 유형입니다.
 - `smoothing_weight`: 모든 패널에 적용되는 스무딩 가중치입니다.
 - `ignore_outliers`: 모든 패널에서 이상치를 무시합니다.
 - `sort_panels_alphabetically`: 모든 섹션에서 패널을 알파벳 순으로 정렬합니다.
 - `group_by_prefix`: 패널을 첫 번째 또는 마지막 접두사로 그룹화합니다(첫 번째 또는 마지막).