---
title: run 필터 및 검색
description: Projects 페이지에서 사이드바와 테이블을 사용하는 방법
menu:
  default:
    identifier: ko-guides-models-track-runs-filter-runs
    parent: what-are-runs
---

프로젝트 페이지를 활용하여 W&B에 로그된 run 에서 인사이트를 얻어보세요. **Workspace** 페이지와 **Runs** 페이지 모두에서 run 을 필터링하고 검색할 수 있습니다.

## run 필터링하기

run 의 상태, [태그]({{< relref path="#filter-runs-with-tags" lang="ko" >}}), [정규 표현식(RegEx)]({{< relref path="#filter-runs-with-regular-expressions-regex" lang="ko" >}}) 또는 기타 속성별로 필터 버튼을 통해 run 을 필터링할 수 있습니다.

run 컬러를 편집, 랜덤화, 리셋하는 방법 등 자세한 내용은 [run 컬러 커스터마이즈]({{< relref path="guides/models/track/runs/run-colors" lang="ko" >}})를 참고하세요.

### 태그로 run 필터링하기

필터 버튼을 사용해 run 의 태그로 필터링할 수 있습니다.

1. 프로젝트 사이드바에서 **Runs** 탭을 클릭하세요.
2. run 테이블 상단에 위치한 깔때기 모양의 **Filter** 버튼을 선택하세요.
3. 왼쪽에서 오른쪽 순으로 드롭다운 메뉴에서 `"Tags"`를 선택하고, 논리 연산자를 선택한 뒤 필터 검색 값을 입력하세요.

### 정규식으로 run 필터링하기

정규식만으로 원하는 결과가 나오지 않으면, [태그]({{< relref path="tags.md" lang="ko" >}})를 활용하여 Runs Table에서 run 을 필터링할 수 있습니다. 태그는 run 생성 시 또는 run 이 종료된 후에도 추가할 수 있습니다. 특정 run 에 태그를 추가했다면, 아래 GIF와 같이 태그 필터를 적용할 수 있습니다.

{{< img src="/images/app_ui/filter_runs.gif" alt="태그별 run 필터링" >}}

1. 프로젝트 사이드바에서 **Runs** 탭을 클릭하세요.
2. run 테이블 상단의 검색 박스를 클릭하세요.
3. **RegEx** 토글(.*)이 활성화되어 있는지 확인하세요(토글이 파란색이어야 합니다).
4. 검색 박스에 원하는 정규 표현식을 입력하세요.

## run 검색하기

정규 표현식(RegEx)을 사용하여 지정한 패턴에 맞는 run 을 찾을 수 있습니다. 검색 박스에 쿼리를 입력하면 워크스페이스의 그래프와 테이블의 행 모두에서 볼 수 있는 run 이 필터링됩니다.

## run 그룹화하기

하나 이상의 컬럼(숨겨진 컬럼 포함) 기준으로 run 을 그룹화할 수 있습니다.

1. 검색 박스 바로 아래의 줄이 그어진 종이 모양의 **Group** 버튼을 클릭하세요.
1. 결과를 그룹화할 하나 이상의 컬럼을 선택하세요.
1. 그룹으로 묶인 run 세트는 기본적으로 접힌 상태입니다. 그룹 이름 옆의 화살표를 클릭하면 펼칠 수 있습니다.

## 최소/최대 값으로 run 정렬하기

로그된 메트릭의 최소값 또는 최대값을 기준으로 run 테이블을 정렬할 수 있습니다. 가장 좋은(또는 나쁜) 값만 보고 싶을 때 매우 유용합니다.

다음 단계에 따라 특정 메트릭의 최소 또는 최대 기록 값 기반으로 run 테이블을 정렬할 수 있습니다:

1. 정렬에 사용할 메트릭이 있는 컬럼 위에 마우스를 올려두세요.
2. 세로로 세 줄이 있는 메뉴(케밥 메뉴)를 선택하세요.
3. 드롭다운에서 **Show min** 또는 **Show max** 중 하나를 선택하세요.
4. 같은 드롭다운에서 **Sort by asc** 또는 **Sort by desc**를 선택해 오름차순 또는 내림차순으로 정렬할 수 있습니다.

{{< img src="/images/app_ui/runs_min_max.gif" alt="최소/최대 값별 정렬" >}}

## run 의 종료 시간 검색

클라이언트 프로세스에서 마지막 하트비트를 기록하는 `End Time` 이라는 컬럼을 제공합니다. 이 필드는 기본적으로 숨겨져 있습니다.

{{< img src="/images/app_ui/search_run_endtime.png" alt="End Time 컬럼" >}}

## run 테이블을 CSV로 내보내기

모든 run, 하이퍼파라미터, 요약 메트릭 테이블을 다운로드 버튼으로 CSV 파일로 내보낼 수 있습니다.

{{< img src="/images/app_ui/export_to_csv.gif" alt="CSV 내보내기 미리보기 모달" >}}