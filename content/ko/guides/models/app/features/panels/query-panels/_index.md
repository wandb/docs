---
title: 쿼리 패널
description: 이 페이지의 일부 기능은 베타 버전으로, 기능 플래그로 숨겨져 있습니다. 프로필 페이지의 자기소개에 `weave-plot`을
  추가하면 관련 모든 기능을 사용할 수 있습니다.
cascade:
- url: guides/app/features/panels/query-panels/:filename
menu:
  default:
    identifier: ko-guides-models-app-features-panels-query-panels-_index
    parent: panels
url: guides/app/features/panels/query-panels
---

{{% alert %}}
W&B Weave를 찾으시나요? 생성형 AI 애플리케이션 빌드를 위한 W&B의 툴 모음에 대해 궁금하다면 weave 문서를 여기에서 확인하세요: [wandb.me/weave](https://wandb.github.io/weave/?utm_source=wandb_docs&utm_medium=docs&utm_campaign=weave-nudge).
{{% /alert %}}

쿼리 패널을 사용해 데이터를 쿼리하고, 인터랙티브하게 시각화하세요.

{{< img src="/images/weave/pretty_panel.png" alt="Query panel" >}}

## 쿼리 패널 생성하기

쿼리를 워크스페이스나 리포트 내에 추가할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="Project workspace" value="workspace" %}}

  1. 프로젝트의 워크스페이스로 이동하세요.
  2. 우측 상단 코너에서 `Add panel`을 클릭하세요.
  3. 드롭다운 메뉴에서 `Query panel`을 선택하세요.
  {{< img src="/images/weave/add_weave_panel_workspace.png" alt="Add panel dropdown" >}}

{{% /tab %}}

{{% tab header="W&B Report" value="report" %}}

`/Query panel`을 입력 후 선택하세요.

{{< img src="/images/weave/add_weave_panel_report_1.png" alt="Query panel option" >}}

또 다른 방법으로, 여러 run을 쿼리와 연결할 수도 있습니다:
1. 리포트에서 `/Panel grid`를 입력하고 선택하세요.
2. `Add panel` 버튼을 클릭하세요.
3. 드롭다운에서 `Query panel`을 선택하세요.

{{% /tab %}}
{{< /tabpane >}}

## 쿼리 구성 요소

### Expressions

쿼리 expressions를 활용해 W&B에 저장된 run, artifact, model, 테이블 등 다양한 데이터에 쿼리를 수행할 수 있습니다.

#### 예시: 테이블 쿼리하기
예를 들어, W&B Table을 쿼리하고 싶다고 가정해봅시다. 트레이닝 코드에서 `"cifar10_sample_table"`이라는 테이블을 로그했다고 할 때:

```python
import wandb
with wandb.init() as run:
  run.log({"cifar10_sample_table":<MY_TABLE>})
```

쿼리 패널 내에서 아래와 같이 테이블을 쿼리할 수 있습니다:
```python
runs.summary["cifar10_sample_table"]
```
{{< img src="/images/weave/basic_weave_expression.png" alt="Table query expression" >}}

각 부분을 살펴보면:

* `runs`는 쿼리 패널이 워크스페이스 내에 있을 때 자동으로 주입되는 변수로, 해당 워크스페이스에서 보이는 run들의 리스트입니다. [run 내에서 사용 가능한 다양한 속성에 대해서는 여기에서 자세히 확인할 수 있습니다]({{< relref path="../../../../track/public-api-guide.md#understanding-the-different-attributes" lang="ko" >}}).
* `summary`는 Run의 Summary 오브젝트를 반환하는 op입니다. op는 _매핑_되어, 리스트의 각 Run에 적용되어 Summary 오브젝트 리스트가 생성됩니다.
* `["cifar10_sample_table"]`는 대괄호로 표기된 Pick op로, 파라미터로 `predictions`가 들어갑니다. Summary 오브젝트는 딕셔너리 혹은 맵처럼 동작하므로, 각 Summary 오브젝트에서 `predictions` 필드를 선택합니다.

쿼리를 인터랙티브하게 작성하는 방법은 [Query panel demo](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr)에서 자세히 확인할 수 있습니다.

### Configurations

패널 좌측 상단의 설정(기어) 아이콘을 클릭하면 쿼리 설정 창이 확장됩니다. 여기선 패널 타입과 결과 패널의 파라미터를 설정할 수 있습니다.

{{< img src="/images/weave/weave_panel_config.png" alt="Panel configuration menu" >}}

### Result panels

마지막으로, 쿼리 결과 패널은 쿼리 expression의 결과를 시각화해 줍니다. 쿼리 패널에서 설정한 방식에 따라 데이터를 인터랙티브하게 보여줍니다. 아래 이미지는 동일한 데이터를 Table과 Plot 형태로 보여주는 예시입니다.

{{< img src="/images/weave/result_panel_table.png" alt="Table result panel" >}}

{{< img src="/images/weave/result_panel_plot.png" alt="Plot result panel" >}}

## 기본 연산
쿼리 패널에서는 다음과 같은 일반적인 연산을 할 수 있습니다.

### Sort
열 옵션에서 정렬할 수 있습니다:
{{< img src="/images/weave/weave_sort.png" alt="Column sort options" >}}

### Filter
쿼리 내에서 직접 필터링하거나, 좌측 상단의 필터 버튼(두 번째 이미지)을 사용할 수 있습니다.
{{< img src="/images/weave/weave_filter_1.png" alt="Query filter syntax" >}}
{{< img src="/images/weave/weave_filter_2.png" alt="Filter button" >}}

### Map
Map 연산은 리스트를 순회하며 각 요소에 함수를 적용합니다. 패널 쿼리로 직접 하거나, 열 옵션에서 새로운 컬럼을 추가해 사용할 수 있습니다.
{{< img src="/images/weave/weave_map.png" alt="Map operation query" >}}
{{< img src="/images/weave/weave_map.gif" alt="Map column insertion" >}}

### Groupby
쿼리 또는 열 옵션에서 Groupby 연산이 가능합니다.
{{< img src="/images/weave/weave_groupby.png" alt="Group by query" >}}
{{< img src="/images/weave/weave_groupby.gif" alt="Group by column options" >}}

### Concat
concat 연산으로 두 개의 테이블을 이어 붙이거나, 패널 설정에서 concat 또는 join을 할 수 있습니다.
{{< img src="/images/weave/weave_concat.gif" alt="Table concatenation" >}}

### Join
쿼리에서 직접 테이블을 join하는 것도 가능합니다. 아래는 join 쿼리 예시입니다:
```python
project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat.join(\
project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat,\
(row) => row["Label"],(row) => row["Label"], "Table1", "Table2",\
"false", "false")
```
{{< img src="/images/weave/weave_join.png" alt="Table join operation" >}}

왼쪽 테이블은 아래 쿼리로 생성됩니다:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_0"].table.rows.concat.join
```
오른쪽 테이블은 다음 쿼리로 생성됩니다:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_1"].table.rows.concat
```
각 부분 설명:
* `(row) => row["Label"]`는 각 테이블에서 join할 컬럼을 지정하는 선택자입니다.
* `"Table1"`과 `"Table2"`는 join 후 각 테이블의 이름을 지정합니다.
* `true`와 `false`는 left/right inner 및 outer join 설정에 사용됩니다.

## Runs object
쿼리 패널을 통해 `runs` 오브젝트에 엑세스할 수 있습니다. Run 오브젝트는 실험의 레코드를 저장합니다. 자세한 내용은 [Accessing runs object](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object)에서 확인하세요. 간단히 정리하면, `runs` 오브젝트는 다음 정보를 제공합니다:
* `summary`: run 결과를 요약한 정보의 딕셔너리입니다. 정확도, 손실값 등의 스칼라 값이나 대용량 파일이 저장됩니다. 기본적으로 `wandb.Run.log()`는 마지막으로 기록된 시계열 데이터를 summary에 저장합니다. 직접 summary 내용을 설정할 수도 있습니다. summary는 run의 "출력"이라 생각하시면 됩니다.
* `history`: 트레이닝 중 변화하는 값(예: loss 등)을 저장하는 딕셔너리들의 리스트입니다. `wandb.Run.log()` 명령으로 이 오브젝트에 값이 추가됩니다.
* `config`: run의 설정값, 즉 트레이닝 run의 하이퍼파라미터 또는 데이터셋 artifact 생성 run의 전처리 방법 등이 딕셔너리 형태로 저장됩니다. 이는 run의 "입력"이라 보면 됩니다.
{{< img src="/images/weave/weave_runs_object.png" alt="Runs object structure" >}}

## Artifacts 엑세스

Artifacts는 W&B의 핵심 개념 중 하나입니다. 버전이 명확히 정의되고 이름이 지정된 파일/디렉토리의 모음입니다. Artifacts를 사용해 모델 가중치, 데이터셋, 기타 파일/디렉토리도 추적할 수 있습니다. Artifacts는 W&B에 저장되며, 다운로드하거나 다른 run에서 사용할 수 있습니다. 자세한 예시는 [Accessing artifacts](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts)에서 확인하세요. 일반적으로 Artifacts는 `project` 오브젝트에서 접근할 수 있습니다:
* `project.artifactVersion()`: 해당 프로젝트 내에서 이름과 버전에 따라 특정 artifact 버전을 반환합니다.
* `project.artifact("")`: 해당 프로젝트 내에서 지정한 이름의 artifact를 반환합니다. 이후 `.versions`를 사용해 모든 버전의 리스트를 가져올 수 있습니다.
* `project.artifactType()`: 프로젝트 내에서 지정된 이름의 `artifactType`을 반환합니다. 이후 `.artifacts`를 사용해 해당 타입의 artifact 리스트를 가져올 수 있습니다.
* `project.artifactTypes`: 프로젝트에 속한 모든 artifact 타입의 리스트를 반환합니다.
{{< img src="/images/weave/weave_artifacts.png" alt="Artifact access methods" >}}