---
title: Filter and search runs
description: 프로젝트 페이지에서 사이드바 및 테이블을 사용하는 방법
menu:
  default:
    identifier: ko-guides-models-track-runs-filter-runs
    parent: what-are-runs
---

W&B에 기록된 run으로부터 얻은 통찰력을 프로젝트 페이지에서 활용하세요. **Workspace** 페이지와 **Runs** 페이지 모두에서 run을 필터링하고 검색할 수 있습니다.

## Run 필터링

필터 버튼을 사용하여 상태, 태그 또는 기타 속성을 기준으로 run을 필터링합니다.

### 태그로 Run 필터링

필터 버튼을 사용하여 태그를 기준으로 run을 필터링합니다.

{{< img src="/images/app_ui/filter_runs.gif" alt="" >}}

### 정규식으로 Run 필터링

정규식으로 원하는 결과를 얻을 수 없는 경우, [태그]({{< relref path="tags.md" lang="ko" >}})를 사용하여 Runs Table에서 run을 필터링할 수 있습니다. 태그는 run 생성 시 또는 완료 후에 추가할 수 있습니다. 태그가 run에 추가되면 아래 GIF와 같이 태그 필터를 추가할 수 있습니다.

{{< img src="/images/app_ui/tags.gif" alt="If regex doesn't provide you the desired results, you can make use of tags to filter out the runs in Runs Table" >}}

## Run 검색

 regex 를 사용하여 지정한 정규식으로 run을 찾습니다. 검색 상자에 쿼리를 입력하면 Workspace의 그래프에서 보이는 run과 테이블의 행이 필터링됩니다.

## Run 그룹화

숨겨진 열을 포함하여 하나 이상의 열을 기준으로 run을 그룹화하려면 다음을 수행합니다.

1. 검색 상자 아래에 있는 줄이 그어진 종이 모양의 **Group** 버튼을 클릭합니다.
2. 결과를 그룹화할 열을 하나 이상 선택합니다.
3. 그룹화된 run 세트는 기본적으로 축소됩니다. 확장하려면 그룹 이름 옆에 있는 화살표를 클릭합니다.

## 최소값 및 최대값으로 Run 정렬

기록된 메트릭의 최소값 또는 최대값으로 run 테이블을 정렬합니다. 이는 가장 좋거나 가장 나쁜 기록 값을 보려는 경우에 특히 유용합니다.

다음 단계에서는 기록된 최소값 또는 최대값을 기준으로 특정 메트릭으로 run 테이블을 정렬하는 방법을 설명합니다.

1. 정렬하려는 메트릭이 있는 열 위에 마우스 커서를 올립니다.
2. 케밥 메뉴(세 개의 세로선)를 선택합니다.
3. 드롭다운에서 **Show min** 또는 **Show max**를 선택합니다.
4. 동일한 드롭다운에서 **Sort by asc** 또는 **Sort by desc**를 선택하여 각각 오름차순 또는 내림차순으로 정렬합니다.

{{< img src="/images/app_ui/runs_min_max.gif" alt="" >}}

## Run에 대한 종료 시간 검색

클라이언트 프로세스에서 마지막 heartbeat를 기록하는 `End Time`이라는 열을 제공합니다. 이 필드는 기본적으로 숨겨져 있습니다.

{{< img src="/images/app_ui/search_run_endtime.png" alt="" >}}

## Run 테이블을 CSV로 내보내기

다운로드 버튼을 사용하여 모든 run, 하이퍼파라미터 및 요약 메트릭 테이블을 CSV로 내보냅니다.

{{< img src="/images/app_ui/export_to_csv.gif" alt="" >}}
