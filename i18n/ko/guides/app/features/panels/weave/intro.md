---
description: Some features on this page are in beta, hidden behind a feature flag.
  Add `weave-plot` to your bio on your profile page to unlock all related features.
slug: /guides/app/features/panels/weave
displayed_sidebar: default
---

# Weave

## 도입

Weave 패널은 사용자가 W&B에서 데이터를 직접 쿼리하고, 결과를 시각화하며, 상호 작용적으로 더 깊게 분석할 수 있게 해줍니다.

![](/images/weave/pretty_panel.png)

:::tip
이 팀이 벤치마크를 시각화하기 위해 Weave 패널을 어떻게 사용했는지 보려면 [이 리포트](http://wandb.me/keras-xla-benchmark)를 참조하세요.
:::

## Weave 패널 만들기

Weave 패널을 추가하려면:

* 워크스페이스에서 `패널 추가`를 클릭하고 `Weave`를 선택하세요.
![](/images/weave/add_weave_panel_workspace.png)
* 리포트에서:
  * `/weave`를 입력하고 `Weave`를 선택하여 독립적인 Weave 패널을 추가하세요.
  ![](/images/weave/add_weave_panel_report_1.png)
  * `/패널 그리드` -> `패널 그리드`를 입력한 다음 `패널 추가` -> `Weave`를 클릭하여 실행 세트와 연관된 Weave 패널을 추가하세요.
  ![](/images/weave/add_weave_panel_report_2.png)

## 구성 요소

### Weave 식

Weave 식은 사용자가 W&B에 저장된 데이터를 쿼리할 수 있게 해줍니다 - 실행부터 아티팩트, 모델, 테이블 등에 이르기까지. `wandb.log({"cifar10_sample_table":<MY_TABLE>})`으로 테이블을 로그할 때 생성할 수 있는 일반적인 Weave 식:

![](/images/weave/basic_weave_expression.png)

이를 분해해보겠습니다:

* `runs`는 워크스페이스에 있는 Weave 패널 식에서 자동으로 주입되는 변수입니다. 그 "값"은 해당 워크스페이스에서 볼 수 있는 실행 목록입니다. [실행 내에서 사용할 수 있는 다양한 속성에 대해 여기서 읽어보세요](../../../../track/public-api-guide.md#understanding-the-different-attributes).
* `summary`는 실행의 요약 객체를 반환하는 op입니다. 주의: ops는 "매핑"되며, 이 op는 목록의 각 실행에 적용되어 요약 객체 목록을 생성합니다.
* `["cifar10_sample_table"]`는 Pick op(대괄호로 표시됨)로, "predictions"의 파라미터를 가집니다. 요약 객체가 사전이나 맵처럼 작동하기 때문에, 이 연산은 각 요약 객체에서 "predictions" 필드를 "선택"합니다.

상호 작용적으로 자체 쿼리를 작성하는 방법을 배우려면 [이 리포트](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr)를 참조하세요. 이 리포트는 Weave에서 사용할 수 있는 기본 연산부터 데이터의 다른 고급 시각화에 이르기까지 다룹니다.

### Weave 설정

패널의 왼쪽 상단 모서리에 있는 기어 아이콘을 선택하여 Weave 설정을 확장하세요. 이를 통해 사용자는 패널의 유형과 결과 패널의 파라미터를 구성할 수 있습니다.

![](/images/weave/weave_panel_config.png)

### Weave 결과 패널

마지막으로, Weave 결과 패널은 설정된 Weave 패널을 사용하여 Weave 식의 결과를 렌더링하고, 데이터를 상호 작용적인 형태로 표시합니다. 다음 이미지는 동일한 데이터의 테이블과 플롯을 보여줍니다.

![](/images/weave/result_panel_table.png)

![](/images/weave/result_panel_plot.png)

## 기본 연산

### 정렬
열 옵션에서 쉽게 정렬할 수 있습니다
![](/images/weave/weave_sort.png)

### 필터
직접 쿼리에서 필터링하거나 상단 왼쪽 모서리의 필터 버튼(두 번째 이미지)을 사용할 수 있습니다
![](/images/weave/weave_filter_1.png)
![](/images/weave/weave_filter_2.png)

### 맵
맵 연산은 리스트를 순회하며 데이터의 각 요소에 함수를 적용합니다. Weave 쿼리를 직접 사용하거나 열 옵션에서 새 열을 삽입하여 이를 수행할 수 있습니다.
![](/images/weave/weave_map.png)
![](/images/weave/weave_map.gif)

### 그룹화
쿼리를 사용하거나 열 옵션에서 그룹화할 수 있습니다.
![](/images/weave/weave_groupby.png)
![](/images/weave/weave_groupby.gif)

### 연결
연결 연산을 사용하면 2개의 테이블을 연결하고 패널 설정에서 연결하거나 조인할 수 있습니다
![](/images/weave/weave_concat.gif)

### 조인
또한 쿼리에서 직접 테이블을 조인할 수 있습니다, 여기서:
* `project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat`는 첫 번째 테이블입니다
* `project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat`는 두 번째 테이블입니다
* `(row) => row["Label"]`은 각 테이블의 선택자로, 조인할 열을 결정합니다
* `"Table1"`과 `"Table2"`는 조인될 때 각 테이블의 이름입니다
* `true`와 `false`는 왼쪽 및 오른쪽 내부/외부 조인 설정입니다
![](/images/weave/weave_join.png)

## 실행 오브젝트
다른 것들 중에서, Weave를 사용하면 `runs` 오브젝트에 액세스할 수 있으며, 여기서 실험의 자세한 기록을 저장합니다. [이 리포트](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object)의 섹션에서 더 자세한 내용을 찾을 수 있지만, 간략하게 `runs` 오브젝트는 다음을 사용할 수 있습니다:
* `summary`: 실행 결과를 요약하는 정보의 사전입니다. 이는 정확도 및 손실과 같은 스칼라 또는 큰 파일일 수 있습니다. 기본적으로, `wandb.log()`는 로그된 시계열의 최종 값을 요약으로 설정합니다. 요약의 내용을 직접 설정할 수 있습니다. 요약을 실행의 출력으로 생각하세요.
* `history`: 모델 트레이닝 중에 변하는 값, 예를 들어 손실과 같은 값을 저장하기 위한 사전 목록입니다. `wandb.log()` 명령어는 이 오브젝트에 추가합니다.
* `config`: 트레이닝 실행의 하이퍼파라미터 또는 데이터셋 아티팩트를 생성하는 실행의 전처리 방법과 같은 실행의 구성 정보 사전입니다. 이를 실행의 "입력"으로 생각하세요.
![](/images/weave/weave_runs_object.png)

## 아티팩트 엑세스

아티팩트는 W&B의 핵심 개념입니다. 아티팩트는 버전이 지정되고 이름이 지정된 파일 및 디렉토리의 모음입니다. 모델 가중치, 데이터셋 및 기타 파일 또는 디렉토리를 추적하기 위해 아티팩트를 사용하세요. 아티팩트는 W&B에 저장되며, 다른 실행에서 다운로드하거나 사용할 수 있습니다. [이 리포트](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts)의 섹션에서 더 자세한 내용과 예제를 찾을 수 있습니다. 아티팩트는 일반적으로 `project` 오브젝트에서 액세스됩니다:
* `project.artifactVersion()`: 프로젝트 내에서 주어진 이름과 버전에 대한 특정 아티팩트 버전을 반환합니다
* `project.artifact("")`: 프로젝트 내에서 주어진 이름에 대한 아티팩트를 반환합니다. 그런 다음 `.versions`를 사용하여 이 아티팩트의 모든 버전 목록을 얻을 수 있습니다
* `project.artifactType()`: 프로젝트 내에서 주어진 이름에 대한 `artifactType`을 반환합니다. 그런 다음 `.artifacts`를 사용하여 이 유형의 모든 아티팩트 목록을 얻을 수 있습니다
* `project.artifactTypes`: 프로젝트 하에 있는 모든 아티팩트 유형의 목록을 반환합니다
![](/images/weave/weave_artifacts.png)