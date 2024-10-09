---
title: Manage tags
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

태그는 로그된 메트릭이나 아티팩트 데이터로부터 명확하지 않을 수 있는 특정 기능으로 runs에 레이블을 붙이는 데 사용될 수 있습니다 -- 이 run의 모델은 `in_production`, 저 run은 `preemptible`, 이 run은 `baseline`을 나타냅니다.

## 태그 추가 방법

run이 생성될 때 태그를 추가할 수 있습니다: `wandb.init(tags=["tag1", "tag2"])` .

트레이닝 중에도 run의 태그를 업데이트할 수 있습니다 (예: 특정 메트릭이 사전 정의된 기준을 넘었을 경우).

```python
run = wandb.init(entity="entity", project="capsules", tags=["debug"])

...

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```

runs가 W&B에 로그된 후 태그를 추가하는 여러 가지 방법이 있습니다.

<Tabs
  defaultValue="publicapi"
  values={[
    {label: 'Using the Public API', value: 'publicapi'},
    {label: 'Project Page', value: 'projectpage'},
    {label: 'Run Page', value: 'runpage'},
  ]}>
  <TabItem value="publicapi">

run이 생성된 후, [공개 API](../../../guides/track/public-api-guide.md)를 사용하여 태그를 업데이트할 수 있습니다:

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # 여기에서 run 데이터에 기반하여 태그를 선택할 수 있습니다.
run.update()
```

공개 API를 사용하는 방법에 대한 자세한 정보는 [참고 문서](../../../ref/README.md)나 [가이드](../../../guides/track/public-api-guide.md)를 참조하세요.

  </TabItem>
  <TabItem value="projectpage">

이 방법은 동일한 태그 또는 태그들을 다수의 runs에 태그를 붙이는 데 가장 적합합니다.

[Project Page](../pages/project-page.md)에서 [runs 사이드바](../pages/project-page.md#search-for-runs)에서 우측 상단의 테이블 아이콘을 클릭하세요. 이것은 사이드바를 전체 [runs 테이블](runs-table.md)로 확장합니다.

테이블의 run 위로 마우스를 올리면 좌측에 체크박스가 보이거나 모든 runs를 선택할 수 있는 헤더 행에 체크박스가 보입니다.

체크박스를 클릭하여 일괄 작업을 가능하게 하세요. 태그를 적용하고자 하는 runs를 선택하세요.

runs 행 위의 태그 버튼을 클릭하세요.

추가하고자 하는 태그를 입력하고 텍스트 상자 아래의 "Add"를 클릭하여 새 태그를 추가하세요.

  </TabItem>
  <TabItem value="runpage">

이 방법은 단일 run에 수작업으로 태그를 적용하는 데 가장 적합합니다.

[Run Page](../pages/run-page.md)의 좌측 사이드바에서 상단의 [Overview 탭](../pages/run-page.md#overview-tab)을 클릭하세요.

"Tags" 옆의 회색 ➕ 버튼을 클릭하세요. 플러스를 클릭하여 태그를 추가하세요.

추가하고자 하는 태그를 입력하고 텍스트 상자 아래의 "Add"를 클릭하여 새 태그를 추가하세요.

  </TabItem>
</Tabs>

## 태그 제거 방법

UI를 통해 runs에서 태그를 제거할 수 있습니다.

<Tabs
  defaultValue="projectpage"
  values={[
    {label: 'Project Page', value: 'projectpage'},
    {label: 'Run Page', value: 'runpage'},
  ]}>
  <TabItem value="projectpage">

이 방법은 다수의 runs에서 태그를 제거하는 데 가장 적합합니다.

[Project Page](../pages/project-page.md)에서 [runs 사이드바](../pages/project-page.md#search-for-runs)에서 우측 상단의 테이블 아이콘을 클릭하세요. 이것은 사이드바를 전체 [runs 테이블](runs-table.md)로 확장합니다.

테이블의 run 위로 마우스를 올리면 좌측에 체크박스가 보이거나 모든 runs를 선택할 수 있는 헤더 행에 체크박스가 보입니다.

체크박스 중 하나를 클릭하여 일괄 작업을 가능하게 하세요. 태그를 제거하고자 하는 runs를 선택하세요.

runs 행 위의 태그 버튼을 클릭하세요.

태그 옆의 체크박스를 클릭하여 run에서 제거합니다.

  </TabItem>
  <TabItem value="runpage">

[Run Page,](../pages/run-page.md)의 좌측 사이드바에서 상단의 [Overview 탭](../pages/run-page.md#overview-tab)을 클릭하세요. run의 태그는 여기에서 볼 수 있습니다.

태그 위로 마우스를 올리고 "x"를 클릭하여 run에서 제거합니다.

  </TabItem>
</Tabs>