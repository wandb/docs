---
title: run 메트릭 비교
description: 여러 runs 간 메트릭 비교
menu:
  default:
    identifier: ko-guides-models-app-features-panels-run-comparer
    parent: panels
weight: 70
---

Run Comparer 를 사용하여 프로젝트 내 여러 run 의 차이점과 유사점을 한눈에 비교할 수 있습니다.

## Run Comparer 패널 추가하기

1. 페이지 오른쪽 상단에 있는 **Add panels** 버튼을 클릭하세요.
1. **Evaluation** 섹션에서 **Run comparer** 를 선택하세요.

## Run Comparer 사용법
Run Comparer 는 프로젝트에서 처음으로 보이는 10개의 run 에 대해 설정과 로그된 메트릭을 한 컬럼씩 보여줍니다.

- 비교할 run 을 변경하려면, 왼쪽에 있는 run 목록에서 검색, 필터, 그룹, 혹은 정렬을 이용하세요. Run Comparer 는 자동으로 업데이트됩니다.
- 설정 키를 필터하거나 검색하려면, Run Comparer 상단의 검색 필드를 사용하세요.
- 차이점만 빠르게 보고 동일한 값은 숨기려면, 패널 상단에서 **Diff only** 를 토글하세요.
- 컬럼 너비나 행 높이를 조정하려면, 패널 상단의 포맷팅 버튼을 사용하세요.
- 설정값이나 메트릭의 값을 복사하려면, 마우스를 해당 값 위에 올리고 복사 버튼을 클릭하세요. 화면에 표시되지 않을 만큼 길더라도 전체 값을 복사할 수 있습니다.

{{% alert %}}
기본적으로 Run Comparer 는 [`job_type`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}) 값이 다른 run 도 구분하지 않습니다. 즉, 실제로 비교 대상이 아니어도 동일한 프로젝트 내에서 서로 다른 run 들을 비교할 수 있습니다. 예를 들어, 트레이닝 run 과 모델 평가 run 을 비교할 수도 있습니다. 트레이닝 run 에는 run 로그, 하이퍼파라미터, 트레이닝 손실 메트릭, 모델 등이 포함될 수 있습니다. 모델 평가 run 은 해당 모델을 사용하여 새로운 트레이닝 데이터에서 모델 성능을 확인하는 용도로 사용할 수 있습니다.

Runs Table 에서 run 목록을 검색, 필터, 그룹, 또는 정렬하면, Run Comparer 는 처음 10개의 run 을 자동으로 비교합니다. 유사한 run 만 비교하려면 Runs Table 에서 `job_type` 등으로 필터하거나 정렬해서 비교하세요. [run 필터링 방법]({{< relref path="/guides/models/track/runs/filter-runs.md" lang="ko" >}})에 대해 더 알아보세요.
{{% /alert %}}