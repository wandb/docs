---
title: Use the project page
description: 프로젝트 페이지에서 사이드바와 표를 사용하는 방법
displayed_sidebar: default
---

## Runs Table

프로젝트 페이지에서, 실행(run)들은 사이드바에 표시됩니다. 사이드바를 확장하여 하이퍼파라미터와 실행 간의 요약 메트릭 테이블을 볼 수 있습니다.

## Search run names

테이블에서 실행 이름에 대해 전체 [정규 표현식](https://dev.mysql.com/doc/refman/8.0/en/regexp.html) 검색을 지원합니다. 검색 상자에 쿼리를 입력하면 워크스페이스의 그래프에서 볼 수 있는 실행과 테이블의 행이 필터링됩니다.

## Filter and sort runs by minimum and maximum values
로그된 메트릭의 최소값 또는 최대값으로 실행 테이블을 정렬하세요. 이것은 기록된 최상의 (또는 최악의) 값을 보고 싶을 때 특히 유용합니다.

다음 단계는 최소값 또는 최대 기록값을 기준으로 특정 메트릭으로 실행 테이블을 정렬하는 방법을 설명합니다:

1. 정렬하려는 메트릭이 있는 열 위에 마우스를 올려 놓으세요.
2. 케밥 메뉴(세 개의 수직선)를 선택하세요.
3. 드롭다운에서 **Show min** 또는 **Show max**를 선택하세요.
4. 같은 드롭다운에서 **Sort by asc** 또는 **Sort by desc**를 선택하여 오름차순 또는 내림차순으로 정렬합니다.

![](/images/app_ui/runs_min_max.gif)

#### What to do in case regex fails?

정규 표현식이 원하는 결과를 제공하지 않는 경우, [tags](tags.md)를 사용하여 Runs Table에서 실행을 필터링할 수 있습니다. 태그는 실행 생성 시 또는 완료 후에 추가할 수 있습니다. 태그가 실행에 추가되면, 아래 gif에 표시된 대로 태그 필터를 추가할 수 있습니다.

![If regex doesn't provide you the desired results, you can make use of tags to filter out the runs in Runs Table](/images/app_ui/tags.gif)

## Search End Time for runs

클라이언트 프로세스로부터 마지막 하트비트를 로그하는 `End Time`이라는 열을 제공합니다. 이 필드는 기본적으로 숨겨져 있습니다.

![](/images/app_ui/search_run_endtime.png)

## Resize the sidebar

프로젝트 페이지에서 그래프에 더 많은 공간을 만들고 싶으신가요? 사이드바 크기를 조정하려면 열 헤더의 가장자리를 클릭하고 드래그하세요. 그래프에서 실행을 켜거나 끌 때 여전히 눈 아이콘을 클릭할 수 있습니다.

## Add sidebar columns

프로젝트 페이지에서 실행은 사이드바에 표시됩니다. 더 많은 열을 표시하려면:

1. 사이드바의 오른쪽 상단에 있는 버튼을 클릭하여 테이블을 확장하세요.
2. 열 헤더에서 드롭다운 메뉴를 클릭하여 열을 고정하세요.
3. 고정된 열은 테이블을 접을 때 사이드바에서 사용할 수 있습니다.

## Bulk select runs

여러 실행을 한 번에 삭제하거나, 실행 그룹에 태그를 지정하세요. 일괄 선택은 실행 테이블을 정리하는 데 더 쉽게 만들어 줍니다.

![](/images/app_ui/howto_bulk_select.gif)

## Select all runs in table

테이블의 왼쪽 상단 모서리에 있는 체크박스를 클릭하고 "Select all runs"을 클릭하여 현재 필터 세트와 일치하는 모든 실행을 선택하세요.

![](/images/app_ui/all_runs_select.gif)

## Move runs between projects

한 프로젝트에서 다른 프로젝트로 실행을 이동하려면:

1. 테이블을 확장합니다.
2. 이동하려는 실행 옆의 체크박스를 클릭합니다.
3. 이동을 클릭하고 대상 프로젝트를 선택합니다.

![](/images/app_ui/howto_move_runs.gif)

## See active runs

실행 이름 옆의 녹색 점을 찾으세요— 이것은 실행이 테이블과 그래프 범례에서 활성 상태임을 나타냅니다.

## Hide uninteresting runs

중단된 실행을 숨기고 싶으신가요? 짧은 실행이 테이블을 가득 채우고 있나요? 그룹 프로젝트에서 본인의 작업만 보고 싶으신가요? 필터를 사용하여 노이즈를 숨기세요. 추천하는 필터는 다음과 같습니다:

* **Show only my work**: 사용자 이름 아래의 실행만을 필터링합니다.
* **Hide crashed runs**: 테이블에서 중단된 실행을 필터링합니다.
* **Duration**: 새로운 필터를 추가하고 "duration"을 선택하여 짧은 실행을 숨깁니다.

![](/images/app_ui/hide_uninsteresting.png)

## Filter runs with tags

태그를 기반으로 실행을 필터링하려면 필터 버튼을 사용하세요.

![](/images/app_ui/filter_runs.gif)

## Filter and delete unwanted runs

테이블을 삭제하려는 실행으로 필터링한 후, 모든 것을 선택하고 삭제 버튼을 눌러 프로젝트에서 제거할 수 있습니다. 실행 삭제는 프로젝트 전체에 영향을 주며, 리포트에서 실행을 삭제하면 프로젝트의 다른 부분에도 반영됩니다.

![](/images/app_ui/filter_unwanted_runs.gif)

## Export runs table to CSV

모든 실행, 하이퍼파라미터, 요약 메트릭 테이블을 다운로드 버튼을 사용하여 CSV 파일로 내보내세요.

![](/images/app_ui/export_to_csv.gif)

## Search columns in the table

테이블 UI 가이드에서 **Columns** 버튼을 사용하여 열을 검색하세요.

![](/images/app_ui/search_columns.gif)