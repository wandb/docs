---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 태그

태그는 로그된 메트릭이나 아티팩트 데이터로부터 명확하지 않은 특정 특징이 있는 실행을 라벨링하는 데 사용할 수 있습니다 -- 이 실행의 모델은 `in_production`이고, 저 실행은 `preemptible`이며, 이 실행은 `baseline`을 나타냅니다.

## 태그 추가 방법

실행이 생성될 때 태그를 추가할 수 있습니다: `wandb.init(tags=["tag1", "tag2"])` .

또한 학습 중에 실행의 태그를 업데이트할 수 있습니다(예: 특정 메트릭이 사전 정의된 임계값을 넘을 경우):

```python
run = wandb.init(entity="entity", project="capsules", tags=["debug"])

...

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```

W&B에 로그된 후 실행에 태그를 추가하는 몇 가지 방법이 있습니다.

<Tabs
  defaultValue="publicapi"
  values={[
    {label: 'Public API 사용하기', value: 'publicapi'},
    {label: '프로젝트 페이지', value: 'projectpage'},
    {label: '실행 페이지', value: 'runpage'},
  ]}>
  <TabItem value="publicapi">

실행이 생성된 후 [우리의 public API](../../../guides/track/public-api-guide.md)를 사용하여 태그를 업데이트할 수 있습니다:

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # 여기서 실행 데이터를 기반으로 태그를 선택할 수 있습니다
run.update()
```

Public API 사용 방법에 대해 더 자세히 알아보려면 [참조 문서](../../../ref/README.md) 또는 [가이드](../../../guides/track/public-api-guide.md)를 읽어보세요.

  </TabItem>
  <TabItem value="projectpage">

이 방법은 동일한 태그 또는 태그들을 대량의 실행에 태깅하기에 가장 적합합니다.

[프로젝트 페이지](../pages/project-page.md)의 [실행 사이드바](../pages/project-page.md#search-for-runs)에서, 오른쪽 상단에 있는 테이블 아이콘을 클릭합니다. 이렇게 하면 사이드바가 전체 [실행 테이블](runs-table.md)로 확장됩니다.

테이블에서 실행을 마우스로 가리키면 왼쪽에 체크박스가 보이거나, 모든 실행을 선택할 수 있는 헤더 행에서 체크박스를 찾을 수 있습니다.

체크박스를 클릭하여 대량 작업을 활성화합니다. 태그를 적용하고자 하는 실행을 선택합니다.

실행 행 위에 있는 태그 버튼을 클릭합니다.

추가하고자 하는 태그를 입력하고 텍스트 박스 아래에 있는 "추가"를 클릭하여 새 태그를 추가합니다.

  </TabItem>
  <TabItem value="runpage">

이 방법은 수동으로 단일 실행에 태그를 적용하는 데 가장 적합합니다.

[실행 페이지](../pages/run-page.md)의 왼쪽 사이드바에서, 상단 [Overview 탭](../pages/run-page.md#overview-tab)을 클릭합니다.

"태그" 옆에는 회색 ➕ 버튼이 있습니다. 이 플러스를 클릭하여 태그를 추가합니다.

추가하고자 하는 태그를 입력하고 텍스트 박스 아래에 있는 "추가"를 클릭하여 새 태그를 추가합니다.

  </TabItem>
</Tabs>

## 태그 제거 방법

UI를 통해서도 실행에서 태그를 제거할 수 있습니다.

<Tabs
  defaultValue="projectpage"
  values={[
    {label: '프로젝트 페이지', value: 'projectpage'},
    {label: '실행 페이지', value: 'runpage'},
  ]}>
  <TabItem value="projectpage">

이 방법은 대량의 실행에서 태그를 제거하기에 가장 적합합니다.

[프로젝트 페이지](../pages/project-page.md)의 [실행 사이드바](../pages/project-page.md#search-for-runs)에서, 오른쪽 상단에 있는 테이블 아이콘을 클릭합니다. 이렇게 하면 사이드바가 전체 [실행 테이블](runs-table.md)로 확장됩니다.

테이블에서 실행을 마우스로 가리키면 왼쪽에 체크박스가 보이거나, 모든 실행을 선택할 수 있는 헤더 행에서 체크박스를 찾을 수 있습니다.

체크박스를 클릭하여 대량 작업을 활성화합니다. 태그를 제거하고자 하는 실행을 선택합니다.

실행 행 위에 있는 태그 버튼을 클릭합니다.

실행에서 제거하고자 하는 태그 옆의 체크박스를 클릭합니다.

  </TabItem>
  <TabItem value="runpage">

[실행 페이지](../pages/run-page.md)의 왼쪽 사이드바에서, 상단 [Overview 탭](../pages/run-page.md#overview-tab)을 클릭합니다. 여기서 실행의 태그를 볼 수 있습니다.

태그 위로 마우스를 올리고 "x"를 클릭하여 실행에서 제거합니다.

  </TabItem>
</Tabs>