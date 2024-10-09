---
title: Query panels
description: 일부 기능은 베타 버전이며, 기능 플래그 뒤에 숨겨져 있습니다. 관련된 모든 기능을 잠금 해제하려면 프로필 페이지의 바이오에 `weave-plot`을 추가하세요.
slug: /guides/app/features/panels/query-panel
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::안내
W&B Weave 를 찾고 계신가요? W&B의 생성적 AI 애플리케이션 빌드 툴 모음을 원하시나요? weave에 대한 문서를 여기에서 찾으세요: [wandb.me/weave](https://wandb.github.io/weave/?utm_source=wandb_docs&utm_medium=docs&utm_campaign=weave-nudge).
:::

쿼리 패널을 사용하여 데이터를 쿼리하고 대화형으로 시각화하세요.

![](/images/weave/pretty_panel.png)

## 쿼리 패널 만들기

워크스페이스나 리포트 내에서 쿼리를 추가하세요.

<Tabs
  defaultValue="workspace"
  values={[
    {label: 'Project workspace', value: 'workspace'},
    {label: 'W&B Report', value: 'report'},
  ]}>
  <TabItem value="report">

1. 프로젝트의 워크스페이스로 이동합니다.
2. 오른쪽 상단에서 `Add panel`을 클릭합니다.
3. 드롭다운에서 `Query panel`을 선택합니다.
![](/images/weave/add_weave_panel_workspace.png)

  </TabItem>
  <TabItem value="workspace">

1. 리포트 내에서 `/Query panel`을 입력하고 선택합니다. 
![](/images/weave/add_weave_panel_report_1.png)

또는 쿼리를 여러 runs 세트에 연결할 수 있습니다.
1. 리포트 내에서 `/Panel grid`을 입력하고 선택하세요.
2. `Add panel` 버튼을 클릭합니다.
3. 드롭다운에서 `Query panel`을 선택합니다.

  </TabItem>
</Tabs>

## 쿼리 구성 요소

### Expressions

W&B에 저장된 runs, Artifacts, 모델, 테이블 등을 쿼리하려면 쿼리 표현식을 사용하세요.

#### 예제: 테이블 쿼리하기
W&B 테이블을 쿼리하려는 경우를 가정해 보겠습니다. 트레이닝 코드 내에서 `"cifar10_sample_table"`이라는 테이블을 로그합니다:

```python
import wandb
wandb.log({"cifar10_sample_table":<MY_TABLE>})
```

쿼리 패널 내에서 테이블을 쿼리할 수 있습니다:
```python
runs.summary["cifar10_sample_table"]
```
![](/images/weave/basic_weave_expression.png)

이를 상세하게 설명하면:

* `runs` 는 워크스페이스 내의 쿼리 패널 표현식에 자동으로 주입되는 변수로, 그 작업공간에 대해 표시되는 runs의 목록입니다. [다른 run 내 속성에 대한 안내서 읽기](../../../../track/public-api-guide.md#understanding-the-different-attributes).
* `summary` 는 Run의 Summary 객체를 반환하는 작업입니다. 참고로: 작업들은 "매핑"되며, 이 작업은 리스트의 각 Run에 적용되어 Summary 객체들의 리스트가 생성됩니다.
* `["cifar10_sample_table"]` 는 "predictions"의 파라미터가 있는 Pick 작업(대괄호로 표시)에 의해, Summary 객체들이 사전 또는 맵처럼 동작하므로, 이 작업은 각 Summary 객체에서 "predictions" 필드를 선택합니다.

자신의 쿼리를 상호작용적으로 작성하는 방법을 배우려면, [이 리포트](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr)를 참조하세요.

### 설정

패널의 왼쪽 상단에 있는 톱니바퀴 아이콘을 선택하여 쿼리 설정을 확장합니다. 이렇게 하면 사용자가 패널 유형과 결과 패널의 파라미터를 설정할 수 있습니다.

![](/images/weave/weave_panel_config.png)

### 결과 패널

마지막으로 쿼리 결과 패널은 선택된 쿼리 패널을 사용하여 쿼리 표현식의 결과를 렌더링하며, 설정을 통해 데이터를 상호작용형 형태로 표시합니다. 다음 이미지는 동일한 데이터의 테이블과 플롯을 보여줍니다.

![](/images/weave/result_panel_table.png)

![](/images/weave/result_panel_plot.png)

## 기본 작업
쿼리 패널 내에서 수행할 수 있는 일반적인 작업은 다음과 같습니다.
### 정렬
열 옵션에서 정렬합니다:
![](/images/weave/weave_sort.png)

### 필터
쿼리 내에서 직접 필터링하거나 왼쪽 상단의 필터 버튼(두 번째 이미지)을 사용하여 필터링할 수 있습니다.
![](/images/weave/weave_filter_1.png)
![](/images/weave/weave_filter_2.png)

### 맵
맵 작업은 리스트를 반복하고 데이터의 각 요소에 기능을 적용합니다. 패널 쿼리 또는 열 옵션에서 새로운 열을 삽입하여 이를 직접 수행할 수 있습니다.
![](/images/weave/weave_map.png)
![](/images/weave/weave_map.gif)

### 그룹화
쿼리나 열 옵션에서 그룹화할 수 있습니다.
![](/images/weave/weave_groupby.png)
![](/images/weave/weave_groupby.gif)

### 연결
연결 작업을 통해 두 개의 테이블을 연결하고 패널 설정에서 연결하거나 조인할 수 있습니다.
![](/images/weave/weave_concat.gif)

### 조인
쿼리에서 직접 테이블을 조인하는 것도 가능합니다. 다음 쿼리 표현식을 고려하세요:
```python
project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat.join(\
project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat,\
(row) => row["Label"],(row) => row["Label"], "Table1", "Table2",\
"false", "false")
```
![](/images/weave/weave_join.png)

왼쪽 테이블은 다음에서 생성됩니다:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_0"].table.rows.concat.join
```
오른쪽 테이블은 다음에서 생성됩니다:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_1"].table.rows.concat
```
여기서:
* `(row) => row["Label"]`은 각 테이블에 대한 선택기로, 어느 열을 기준으로 조인할지를 결정합니다.
* `"Table1"`과 `"Table2"`는 조인되었을 때 각 테이블의 이름입니다.
* `true`와 `false`는 좌측 및 우측 내부/외부 조인 설정에 사용됩니다.

## Runs 오브젝트
`runs` 오브젝트에 엑세스하려면 쿼리 패널을 사용하세요. Run 오브젝트는 실험 기록을 저장합니다. 이에 대한 자세한 내용은 리포트의 [이 섹션](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object)에서 확인할 수 있지만, 간단히 살펴보면, `runs` 오브젝트는 다음과 같이 사용 가능합니다:
* `summary`: Run의 결과를 요약한 정보를 담고 있는 사전입니다. 이는 정확도와 손실과 같은 스칼라 값을 포함하거나 큰 파일일 수 있습니다. 기본적으로 `wandb.log()`는 최종 값의 로그된 시계열을 요약으로 설정합니다. 요약 내용을 직접 설정할 수 있습니다. 요약은 run의 출력물이라 생각하세요.
* `history`: 모델이 트레이닝 중일 때 변하는 값을 저장하기 위한 사전 목록입니다. `wandb.log()` 명령은 이 오브젝트에 추가됩니다.
* `config`: 트레이닝 run의 하이퍼파라미터나 데이터셋 Artifact를 생성하는 run의 전처리 방식을 포함한 run의 설정 정보를 담고 있는 사전입니다. 이를 run의 "입력"이라고 생각하세요.
![](/images/weave/weave_runs_object.png)

## Artifacts 엑세스

Artifacts는 W&B의 핵심 개념입니다. 이는 파일과 디렉토리의 버전화된 조합입니다. Artifacts를 사용하여 모델 웨이트, 데이터셋 및 기타 파일 또는 디렉토리를 추적하세요. Artifacts는 W&B에 저장되며 다른 run에서 다운로드하거나 사용할 수 있습니다. [이 섹션](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts)에 있는 리포트에서 더 많은 세부 사항과 예제를 찾을 수 있습니다. Artifacts는 일반적으로 `project` 오브젝트에서 엑세스됩니다:
* `project.artifactVersion()`: 주어진 이름 및 버전에 대해 특정 Artifact 버전을 반환합니다.
* `project.artifact("")`: 프로젝트 내 주어진 이름에 대해 Artifact를 반환합니다. 이를 통해 `.versions`를 사용하여 해당 Artifact의 모든 버전 목록을 얻을 수 있습니다.
* `project.artifactType()`: 프로젝트 내 주어진 이름에 대한 `artifactType`을 반환합니다. 이를 통해 `.artifacts`를 사용하여 이 유형의 모든 Artifact 목록을 얻을 수 있습니다.
* `project.artifactTypes`: 프로젝트의 모든 Artifact 유형의 목록을 반환합니다.
![](/images/weave/weave_artifacts.png)