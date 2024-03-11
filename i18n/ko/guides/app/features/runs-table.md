---
description: How to use the sidebar and table on the project page
displayed_sidebar: default
---

# Runs Table

프로젝트 페이지에서 우리는 사이드바에서 run을 보여줍니다. 사이드바를 확장하여 표에서 run들의 하이퍼파라미터와 요약 메트릭을 확인하세요.

## run 이름 검색

표에서 run 이름에 대한 전체 [regex](https://dev.mysql.com/doc/refman/8.0/en/regexp.html) 검색을 지원합니다. 검색 상자에 쿼리를 입력하면 워크스페이스의 그래프에 보이는 가시적인 run들을 필터링하고 표의 행을 필터링합니다.

## 최소 및 최대 값으로 run 필터링 및 정렬
로그된 메트릭의 최소 또는 최대 값에 따라 run 표를 정렬합니다. 이는 기록된 최고(또는 최악)의 값을 보고 싶을 때 특히 유용합니다.

특정 메트릭을 기준으로 최소 또는 최대 기록된 값으로 run 표를 정렬하는 방법은 다음과 같습니다:

1. 정렬하고자 하는 메트릭이 있는 열 위에 마우스를 올립니다.
2. 케밥 메뉴(세로 세 줄)를 선택합니다.
3. 드롭다운에서 **Show min** 또는 **Show max**를 선택합니다.
4. 같은 드롭다운에서 **Sort by asc** 또는 **Sort by desc**를 선택하여 오름차순 또는 내림차순으로 정렬합니다.

![](/images/app_ui/runs_min_max.gif)

#### regex가 실패한 경우에는 어떻게 하나요?

regex가 원하는 결과를 제공하지 않는 경우 [tags](tags.md)를 사용하여 Runs Table에서 run들을 필터링할 수 있습니다. 태그는 run 생성 시 또는 완료된 후에 추가할 수 있습니다. run에 태그가 추가되면 아래 gif와 같이 태그 필터를 추가할 수 있습니다.

![regex가 원하는 결과를 제공하지 않는 경우 Runs Table에서 run들을 필터링하기 위해 태그를 사용할 수 있습니다](@site/static/images/app_ui/tags.gif)

## run의 종료 시간 검색

`End Time`이라는 열을 제공하여 클라이언트 프로세스의 마지막 심장박동을 로그합니다. 이 필드는 기본적으로 숨겨져 있습니다.

![](/images/app_ui/search_run_endtime.png)

## 사이드바 크기 조정

프로젝트 페이지의 그래프를 위한 더 많은 공간을 만들고 싶으신가요? 열 헤더의 가장자리를 클릭하고 드래그하여 사이드바의 크기를 조정하세요. 그래프에서 run을 켜고 끄는 눈 아이콘을 여전히 클릭할 수 있습니다.

![](https://downloads.intercomcdn.com/i/o/153755378/d54ae70fb8155657a87545b1/howto+-+resize+column.gif)

## 사이드바 열 추가

프로젝트 페이지에서 사이드바에 run을 보여줍니다. 더 많은 열을 표시하려면:

1. 사이드바의 오른쪽 상단에 있는 버튼을 클릭하여 표를 확장하십시오.
2. 열 헤더에서 드롭다운 메뉴를 클릭하여 열을 고정하십시오.
3. 고정된 열은 표를 접었을 때 사이드바에 표시됩니다.

여기 화면 캡처가 있습니다. 표를 확장하고, 두 열을 고정하고, 표를 접은 다음, 사이드바의 크기를 조정합니다.

![](https://downloads.intercomcdn.com/i/o/152951680/cf8cbc6b35e923be2551ba20/howto+-+pin+rows+in+table.gif)

## run 대량 선택

한 번에 여러 run을 삭제하거나 run 그룹에 태그를 지정하는 등— 대량 선택을 사용하면 run 표를 정리하기가 더 쉽습니다.

![](/images/app_ui/howto_bulk_select.gif)

## 표에서 모든 run 선택

표의 왼쪽 상단에 있는 체크박스를 클릭하고 "Select all runs"를 클릭하여 현재 필터 세트와 일치하는 모든 run을 선택하십시오.

![](/images/app_ui/all_runs_select.gif)

## 프로젝트 간에 run 이동

한 프로젝트에서 다른 프로젝트로 run을 이동하려면:

1. 표를 확장합니다
2. 이동하려는 run 옆의 체크박스를 클릭합니다
3. 이동을 클릭하고 대상 프로젝트를 선택합니다

![](/images/app_ui/howto_move_runs.gif)

## 활성 run 확인

run 이름 옆에 녹색 점이 있는 것을 찾아보세요— 이것은 표와 그래프 범례에서 활성 상태임을 나타냅니다.

## 흥미 없는 run 숨기기

추락한 run을 숨기고 싶으신가요? 짧은 run이 표를 채우고 있나요? 그룹 프로젝트에서 자신의 작업만 보고 싶으신가요? 필터로 노이즈를 숨깁니다. 우리가 추천하는 몇 가지 필터:

* **Show only my work**는 사용자 이름 아래의 run만 필터링합니다
* **Hide crashed runs**는 테이블에서 추락한 것으로 표시된 run을 필터링합니다
* **Duration**: 새 필터를 추가하고 "duration"을 선택하여 짧은 run을 숨깁니다

![](/images/app_ui/hide_uninsteresting.png)

## 태그로 run 필터링

필터 버튼으로 태그를 기반으로 run을 필터링합니다.

![](/images/app_ui/filter_runs.gif)

## 원치 않는 run 필터링 및 삭제

삭제하고 싶은 run만 표에 필터링하면 모두 선택하고 삭제를 눌러 프로젝트에서 제거할 수 있습니다. run을 삭제하는 것은 프로젝트 전체에 영향을 미치므로 리포트에서 run을 삭제하면 프로젝트의 나머지 부분에 반영됩니다.

![](/images/app_ui/filter_unwanted_runs.gif)

## CSV로 run 표 내보내기

다운로드 버튼으로 모든 run, 하이퍼파라미터 및 요약 메트릭의 표를 CSV로 내보냅니다.

![](/images/app_ui/export_to_csv.gif)

## 표에서 열 검색

**Columns** 버튼으로 테이블 UI 가이드에서 열을 검색합니다.

![](/images/app_ui/search_columns.gif)