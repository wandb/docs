---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 태그

태그는 로그된 메트릭이나 아티팩트 데이터에서 명확하지 않은 특정 특징을 가진 run에 라벨을 붙이는 데 사용될 수 있습니다 -- 이 run의 모델이 `in_production`이고, 저 run은 `preemptible`이며, 이 run은 `baseline`을 대표합니다.

## 태그 추가 방법

run이 생성될 때 태그를 추가할 수 있습니다: `wandb.init(tags=["tag1", "tag2"])`.

또한 트레이닝 중에 특정 메트릭이 사전에 정의된 임계값을 넘을 경우와 같이 run의 태그를 업데이트할 수도 있습니다:

```python
run = wandb.init(entity="entity", project="capsules", tags=["debug"])

...

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```

run이 W&B에 로그된 후에도 여러 방법으로 태그를 추가할 수 있습니다.

<Tabs
  defaultValue="publicapi"
  values={[
    {label: 'Public API 사용하기', value: 'publicapi'},
    {label: '프로젝트 페이지', value: 'projectpage'},
    {label: 'Run 페이지', value: 'runpage'},
  ]}>
  <TabItem value="publicapi">

run이 생성된 후에는 [우리의 public API](../../../guides/track/public-api-guide.md)를 사용하여 다음과 같이 태그를 업데이트할 수 있습니다:

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # 여기서 run 데이터를 기반으로 태그를 선택할 수 있습니다
run.update()
```

Public API 사용 방법에 대해서는 [참조 문서](../../../ref/README.md) 또는 [가이드](../../../guides/track/public-api-guide.md)에서 더 자세히 알아볼 수 있습니다.

  </TabItem>
  <TabItem value="projectpage">

이 방법은 동일한 태그 또는 태그들을 대량의 run에 태깅하기에 가장 적합합니다.

[프로젝트 페이지](../pages/project-page.md)의 [runs 사이드바](../pages/project-page.md#search-for-runs)에서 오른쪽 상단의 테이블 아이콘을 클릭합니다. 이렇게 하면 사이드바가 전체 [runs 테이블](runs-table.md)로 확장됩니다.

테이블에서 run 위로 마우스를 올리면 왼쪽에 체크박스가 나타나거나 모든 run을 선택할 수 있는 헤더 행의 체크박스를 볼 수 있습니다.

체크박스를 클릭하여 대량 작업을 활성화합니다. 태그를 적용하고자 하는 run을 선택합니다.

run 행 위에 있는 태그 버튼을 클릭합니다.

추가하고 싶은 태그를 입력하고 텍스트 상자 아래의 "추가"를 클릭하여 새 태그를 추가합니다.

  </TabItem>
  <TabItem value="runpage">

이 방법은 수동으로 단일 run에 태그 또는 태그들을 적용하는 데 가장 적합합니다.

[Run 페이지](../pages/run-page.md)의 왼쪽 사이드바에서 상단의 [Overview 탭](../pages/run-page.md#overview-tab)을 클릭합니다.

"태그" 옆에는 회색 ➕ 버튼이 있습니다. 이 플러스를 클릭하여 태그를 추가합니다.

추가하고 싶은 태그를 입력하고 텍스트 상자 아래의 "추가"를 클릭하여 새 태그를 추가합니다.

  </TabItem>
</Tabs>

## 태그 제거 방법

UI를 통해서도 run에서 태그를 제거할 수 있습니다.

<Tabs
  defaultValue="projectpage"
  values={[
    {label: '프로젝트 페이지', value: 'projectpage'},
    {label: 'Run 페이지', value: 'runpage'},
  ]}>
  <TabItem value="projectpage">

이 방법은 대량의 run에서 태그를 제거하기에 가장 적합합니다.

[프로젝트 페이지](../pages/project-page.md)의 [runs 사이드바](../pages/project-page.md#search-for-runs)에서 오른쪽 상단의 테이블 아이콘을 클릭합니다. 이렇게 하면 사이드바가 전체 [runs 테이블](runs-table.md)로 확장됩니다.

테이블에서 run 위로 마우스를 올리면 왼쪽에 체크박스가 나타나거나 모든 run을 선택할 수 있는 헤더 행의 체크박스를 볼 수 있습니다.

체크박스를 클릭하여 대량 작업을 활성화합니다. 태그를 제거하고자 하는 run을 선택합니다.

run 행 위에 있는 태그 버튼을 클릭합니다.

run에서 제거하려는 태그 옆의 체크박스를 클릭합니다.

  </TabItem>
  <TabItem value="runpage">

[Run 페이지](../pages/run-page.md)의 왼쪽 사이드바에서 상단의 [Overview 탭](../pages/run-page.md#overview-tab)을 클릭합니다. 여기서 run의 태그를 볼 수 있습니다.

태그 위로 마우스를 올리고 "x"를 클릭하여 run에서 해당 태그를 제거합니다.

  </TabItem>
</Tabs>