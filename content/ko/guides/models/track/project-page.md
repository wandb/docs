---
title: 'Projects

  '
description: 모델의 여러 버전을 비교하고, 임시 워크스페이스에서 결과를 탐색하며, 발견한 내용을 리포트로 내보내어 메모와 시각화를 저장하세요.
menu:
  default:
    identifier: ko-guides-models-track-project-page
    parent: experiments
weight: 3
---

*프로젝트*는 결과를 시각화하고, 실험을 비교하며, Artifacts를 보고 다운로드하고, 자동화를 생성하는 등 다양한 작업을 할 수 있는 중앙 공간입니다.

{{% alert %}}
각 프로젝트에는 누가 엑세스할 수 있는지를 결정하는 가시성 설정이 있습니다. 프로젝트에 누가 엑세스할 수 있는지에 대한 자세한 내용은 [프로젝트 가시성]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ko" >}})을 참고하세요.
{{% /alert %}}

각 프로젝트는 다음과 같은 탭으로 구성되어 있습니다:

* [Overview]({{< relref path="project-page.md#overview-tab" lang="ko" >}}): 프로젝트의 스냅샷
* [Workspace]({{< relref path="project-page.md#workspace-tab" lang="ko" >}}): 개인 시각화 샌드박스
* [Runs]({{< relref path="#runs-tab" lang="ko" >}}): 이 프로젝트의 모든 run 목록이 나열된 테이블
* [Automations]({{< relref path="#automations-tab" lang="ko" >}}): 프로젝트에 설정된 자동화 목록
* [Sweeps]({{< relref path="project-page.md#sweeps-tab" lang="ko" >}}): 자동화된 탐색 및 최적화
* [Reports]({{< relref path="project-page.md#reports-tab" lang="ko" >}}): 노트, run, 그래프의 저장된 스냅샷
* [Artifacts]({{< relref path="#artifacts-tab" lang="ko" >}}): 모든 run과 해당 run에 연결된 Artifacts

## Overview 탭

* **Project name**: 프로젝트 이름입니다. W&B는 run을 초기화하면서 지정한 이름을 가진 프로젝트를 자동으로 생성합니다. 우측 상단 **Edit** 버튼을 통해 언제든 프로젝트 이름을 변경할 수 있습니다.
* **Description**: 프로젝트에 대한 설명입니다.
* **Project visibility**: 프로젝트의 가시성. 누가 엑세스할 수 있는지 결정하는 설정입니다. 자세한 내용은 [프로젝트 가시성]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ko" >}})을 참조하세요.
* **Last active**: 최근 이 프로젝트에 데이터가 로그된 시간
* **Owner**: 프로젝트를 소유한 Entity
* **Contributors**: 이 프로젝트에 기여한 User 수
* **Total runs**: 이 프로젝트에서 실행된 전체 run의 수
* **Total compute**: 프로젝트의 모든 run의 시간을 합산한 값
* **Undelete runs**: 드롭다운 메뉴에서 "Undelete all runs"를 클릭하여 삭제된 run을 복구할 수 있습니다.
* **Delete project**: 우측 상단 점 세 개 메뉴에서 프로젝트를 삭제할 수 있습니다.

[실제 예시 보기](https://app.wandb.ai/example-team/sweep-demo/overview)

{{< img src="/images/track/overview_tab_image.png" alt="Project overview tab" >}}


## Workspace 탭

프로젝트의 *workspace*는 실험을 비교해 볼 수 있는 개인 샌드박스입니다. 모델을 구조별로, 하이퍼파라미터별로, 데이터셋, 전처리 방식 등에 따라 그룹화하여 다양한 설정에서 동일한 문제를 해결하는 실험을 조직적으로 관리할 수 있습니다.

**Runs 사이드바**: 프로젝트 내 모든 run 목록을 보여줍니다.

* **Dot 메뉴**: 사이드바의 각 행에 마우스를 올리면 좌측에 메뉴가 나타납니다. 이 메뉴를 사용해 run의 이름 변경, 삭제, 활성 run 중단 등이 가능합니다.
* **Visibility 아이콘**: 눈 모양 아이콘을 클릭하여 그래프에 run을 표시하거나 숨길 수 있습니다.
* **Color**: run의 색상을 프리셋 또는 사용자 지정 색상으로 변경할 수 있습니다.
* **Search**: run 이름으로 검색하여, 해당 이름이 포함된 run만 플롯에 표시됩니다.
* **Filter**: 사이드바 필터를 사용해 표시할 run을 좁힐 수 있습니다.
* **Group**: config 컬럼을 선택하여 run을 동적으로 그룹화할 수 있습니다(예: architecture 기준). 그룹화하면 평균값이 선으로, 분산 구간이 음영처리로 표현됩니다.
* **Sort**: 예를 들어, 가장 낮은 loss나 가장 높은 accuracy로 run을 정렬할 수 있습니다. 정렬된 순서대로 그래프에도 반영됩니다.
* **Expand 버튼**: 사이드바를 전체 테이블로 확장합니다.
* **Run count**: 맨 위에 괄호 속 숫자는 프로젝트 내 전체 run의 개수입니다. (N visualized)는 눈 아이콘이 켜져 실제 그래프에 표시되는 run의 개수입니다. 아래 예시에서는 전체 run이 183개 중 10개만 그래프에 보입니다. 그래프 설정에서 표시할 run 개수를 조정할 수 있습니다.

[Runs 탭](#runs-tab)에서 컬럼을 고정, 숨기기, 순서 변경 등의 설정을 하면 Runs 사이드바에도 똑같이 반영됩니다.

**Panels 레이아웃**: 자유롭게 패널을 추가/삭제하여 결과를 탐색하고, 버전에 따라 다양한 메트릭으로 모델을 비교할 수 있는 공간입니다.

[실제 예시 보기](https://app.wandb.ai/example-team/sweep-demo)

{{< img src="/images/app_ui/workspace_tab_example.png" alt="Project workspace" >}}


### 패널 섹션 추가

섹션 드롭다운 메뉴에서 "Add section"을 클릭해 새로운 패널 섹션을 만들 수 있습니다. 섹션 이름 변경, 순서 이동, 확장/접기 모두 드래그&드롭 또는 버튼 클릭으로 쉽게 할 수 있습니다.

각 섹션의 우측 상단에서 다음과 같은 옵션을 사용할 수 있습니다:

* **Switch to custom layout**: 각 패널을 개별적으로 리사이즈할 수 있는 레이아웃입니다.
* **Switch to standard layout**: 모든 패널을 동시에 리사이즈하고, 페이징 처리가 가능한 레이아웃입니다.
* **Add section**: 드롭다운 또는 페이지 하단의 버튼을 통해 섹션 추가
* **Rename section**: 섹션의 제목을 변경
* **Export section to report**: 이 섹션을 새 리포트로 저장
* **Delete section**: 해당 섹션(내부 그래프 포함) 삭제. 작업영역 하단의 undo 버튼으로 복구 가능
* **Add panel**: 플러스 버튼을 눌러 섹션에 패널 추가

{{< img src="/images/app_ui/add-section.gif" alt="Adding workspace section" >}}

### 패널 섹션 간 이동

패널을 드래그&드롭으로 섹션 간에 자유롭게 이동할 수 있습니다. 패널 우측 상단의 "Move" 버튼을 이용해 이동할 섹션을 선택할 수도 있습니다.

{{< img src="/images/app_ui/move-panel.gif" alt="Moving panels between sections" >}}

### 패널 크기 조절

* **Standard layout**: 모든 패널이 같은 크기를 가지며, 페이지 단위로 보여줍니다. 패널 우하단을 드래그해서 크기 조절, 섹션 우하단을 드래그해서 섹션 크기 조절이 가능합니다.
* **Custom layout**: 각각의 패널을 독립적으로 크기 조절 가능하며, 페이지 구분이 없습니다.

{{< img src="/images/app_ui/resize-panel.gif" alt="Resizing panels" >}}

### 메트릭 검색

워크스페이스 상단의 검색 박스를 사용하여 패널을 필터할 수 있습니다. 이 검색은 기본적으로 패널 타이틀(즉 시각화 중인 메트릭 이름)에 적용됩니다.

{{< img src="/images/app_ui/search_in_the_workspace.png" alt="Workspace search" >}}

## Runs 탭

Runs 탭에서는 run을 필터, 그룹화, 정렬할 수 있습니다.

{{< img src="/images/runs/run-table-example.png" alt="Runs table" >}}

아래 탭에서 Runs 탭에서 자주 사용되는 기능 예시를 소개합니다.

{{< tabpane text=true >}}
   {{% tab header="컬럼 커스터마이즈" %}}
Runs 탭은 run의 다양한 세부정보를 컬럼별로 보여줍니다. 기본적으로 많은 컬럼이 노출됩니다.

{{% alert %}}
Runs 탭에서 커스터마이즈한 설정은 [Workspace 탭]({{< relref path="#workspace-tab" lang="ko" >}})의 **Runs** 셀렉터에도 동일하게 반영됩니다.
{{% /alert %}}

- 모든 컬럼을 보려면 테이블을 가로로 스크롤하세요.
- 컬럼 순서 변경은 컬럼명을 좌우로 드래그하세요.
- 컬럼을 고정하려면, 컬럼명에 마우스를 올리고 액션 메뉴(`...`) 클릭 후 **Pin column** 선택. 고정 컬럼은 **Name** 컬럼 뒤쪽에 위치합니다. 해제하려면 **Unpin column**을 선택하세요.
- 컬럼을 숨기려면, 컬럼명에 마우스를 올리고 액션 메뉴(`...`) 클릭 후 **Hide column** 선택. 숨겨진 컬럼 목록은 **Columns** 클릭 시 확인할 수 있습니다.
- 여러 컬럼을 한 번에 표시/숨김/고정/고정 해제하려면 **Columns**를 클릭하세요.
  - 숨겨진 컬럼을 클릭하면 다시 표시되고,
  - 표시중인 컬럼을 클릭하면 숨겨집니다.
  - 핀 아이콘을 눌러 표시중인 컬럼을 고정할 수 있습니다.

   {{% /tab %}}

   {{% tab header="Sort" %}}
테이블의 특정 컬럼 값으로 모든 행을 정렬할 수 있습니다.

1. 컬럼 타이틀에 마우스를 올리면 케밥 메뉴(세로 점 세 개)가 나타납니다.
2. 케밥 메뉴(세로 점 세 개)를 클릭합니다.
3. **Sort Asc** 또는 **Sort Desc**를 선택해 각각 오름차순, 내림차순 정렬을 적용합니다.

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="Confident predictions" >}}

위 이미지는 `val_acc` 컬럼에서 정렬 옵션을 선택하는 방법을 보여줍니다.
   {{% /tab %}}
   {{% tab header="Filter" %}}
대시보드 좌상단의 **Filter** 버튼을 클릭해 모든 행을 특정 표현식으로 필터링할 수 있습니다.

{{< img src="/images/data_vis/filter.png" alt="Incorrect predictions filter" >}}

**Add filter**를 선택해 하나 이상 필터를 추가할 수 있습니다. 세 개의 드롭다운이 좌→우 순으로 표시되며, 각각 컬럼명, 연산자, 값에 해당합니다.

|                   | Column name | Binary relation    | Value       |
| -----------       | ----------- | ----------- | ----------- |
| Accepted values   | String       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | Integer, float, string, timestamp, null |

표현식 에디터에서 컬럼이름 및 논리 구조에 대해 자동완성으로 옵션을 안내해 줍니다. "and" "or" (필요하다면 괄호)로 여러 논리식을 조합할 수 있습니다.

{{< img src="/images/data_vis/filter_example.png" alt="Filtering runs by validation loss" >}}
위 이미지는 `val_loss` 컬럼 값이 1 이하인 run만 필터링한 예시입니다.
   {{% /tab %}}
   {{% tab header="Group" %}}
특정 컬럼의 값으로 모든 행을 그룹화하려면, 해당 컬럼 헤더에서 **Group by** 버튼을 사용합니다.

{{< img src="/images/data_vis/group.png" alt="Error distribution analysis" >}}

기본적으로는 다른 수치 컬럼을 그룹별 값의 분포를 나타내는 히스토그램으로 볼 수 있습니다. 그룹화는 데이터의 높은 수준 패턴을 이해하는 데 유용합니다.
   {{% /tab %}}
{{< /tabpane >}}



## Automations 탭
Artifacts 버전 관리를 위한 후속 작업을 자동화할 수 있습니다. 자동화 생성을 위해 트리거 이벤트와 작업을 정의하면 됩니다. 작업에는 웹훅 호출, W&B job 실행 등이 포함됩니다. 자세한 내용은 [Automations]({{< relref path="/guides/core/automations/" lang="ko" >}}) 가이드를 참고하세요.

{{< img src="/images/app_ui/automations_tab.png" alt="Automation tab" >}}

## Reports 탭

모든 결과 스냅샷을 한 곳에서 보고, 팀과 발견한 내용을 공유하세요.

{{< img src="/images/app_ui/reports-tab.png" alt="Reports tab" >}}

## Sweeps 탭

프로젝트에서 새로운 [Sweep]({{< relref path="/guides/models/sweeps/" lang="ko" >}})을 시작하세요.

{{< img src="/images/app_ui/sweeps-tab.png" alt="Sweeps tab" >}}

## Artifacts 탭

프로젝트와 연결된 모든 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 확인하세요. 여기에는 트레이닝 데이터셋, [파인튜닝 모델]({{< relref path="/guides/core/registry/" lang="ko" >}}), [메트릭 및 미디어 테이블]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ko" >}}) 등이 포함됩니다.

### Overview 패널

{{< img src="/images/app_ui/overview_panel.png" alt="Artifact overview panel" >}}

Overview 패널에서는 해당 Artifact의 이름, 버전, 중복 방지를 위한 해시값, 생성일, 에일리어스 등의 다양한 주요 정보를 확인할 수 있습니다. 이곳에서 에일리어스를 추가하거나 제거할 수 있고, 특정 버전 및 전체 Artifact에 노트를 남길 수 있습니다.

### Metadata 패널

{{< img src="/images/app_ui/metadata_panel.png" alt="Artifact metadata panel" >}}

Metadata 패널은 Artifact 생성 시 입력한 메타데이터를 제공합니다. 여기에는 Artifact 복원에 필요한 설정 인수, 참고용 URL, run에서 기록된 메트릭 등이 포함될 수 있습니다. 아울러 해당 Artifact를 생성한 run의 설정과 로그 당시의 히스토리 메트릭도 볼 수 있습니다.

### Usage 패널

{{< img src="/images/app_ui/usage_panel.png" alt="Artifact usage panel" >}}

Usage 패널에서는 웹 앱 외부(예: 로컬 머신)에서 사용할 수 있도록 Artifact를 다운로드하는 코드조각을 제공합니다. 이 섹션에서는 해당 Artifact를 생성한 run 및 입력으로 사용하는 run도 함께 확인하고 링크로 엑세스할 수 있습니다.

### Files 패널

{{< img src="/images/app_ui/files_panel.png" alt="Artifact files panel" >}}

Files 패널에는 Artifact와 연결된 파일 및 폴더 목록이 표시됩니다. W&B는 run에서 일부 파일을 자동으로 업로드합니다. 예를 들어, `requirements.txt`는 사용된 라이브러리 버전 정보를, `wandb-metadata.json`, `wandb-summary.json`은 run에 대한 정보를 담고 있습니다. 이외에도 Artifacts 또는 미디어 등 설정에 따라 다른 파일도 업로드됩니다. 파일 트리를 탐색하여 웹 앱 내에서 바로 내용을 확인할 수 있습니다.

[Table]({{< relref path="/guides/models/tables//tables-walkthrough.md" lang="ko" >}})을 Artifacts와 함께 활용하면 더욱 풍부하고 상호작용적인 기능을 경험할 수 있습니다. Table과 Artifacts 활용법은 [여기]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ko" >}})에서 더 확인하세요.

{{< img src="/images/app_ui/files_panel_table.png" alt="Artifact table view" >}}

### Lineage 패널

{{< img src="/images/app_ui/lineage_panel.png" alt="Artifact lineage" >}}

Lineage 패널은 프로젝트에 연결된 모든 Artifact와 그것들을 연결하는 run의 관계를 도식적으로 보여줍니다. run 타입은 블록, Artifact는 원으로 나타나며, 화살표는 어떤 run이 특정 타입의 Artifact를 소비하거나 생성하는지 보여줍니다. 왼쪽 컬럼에서 선택한 Artifact에 해당하는 타입은 강조 표시됩니다.

Explode 토글을 클릭하면 모든 개별 Artifact 버전과 연결된 run의 상세 연결 관계를 볼 수 있습니다.

### Action History Audit 탭

{{< img src="/images/app_ui/action_history_audit_tab_1.png" alt="Action history audit" >}}

{{< img src="/images/app_ui/action_history_audit_tab_2.png" alt="Action history" >}}

Action history audit 탭에서는 컬렉션의 에일리어스 조작 및 멤버십 변경 내역을 모두 확인할 수 있어 해당 리소스의 이력을 추적할 수 있습니다.

### Versions 탭

{{< img src="/images/app_ui/versions_tab.png" alt="Artifact versions tab" >}}

Versions 탭에서는 Artifact의 모든 버전과 해당 버전을 남길 때의 Run History 내 모든 수치 값이 컬럼으로 함께 표시됩니다. 이를 통해 성능 비교 및 관심 버전을 빠르게 식별할 수 있습니다.

## 프로젝트 생성하기
W&B App에서 또는 프로그래밍 코드로 `wandb.init()`를 호출할 때 프로젝트를 지정하여 프로젝트를 만들 수 있습니다.

{{< tabpane text=true >}}
   {{% tab header="W&B App" %}}
W&B App에서는 **Projects** 페이지 또는 팀 랜딩페이지에서 새 프로젝트를 만들 수 있습니다.

**Projects** 페이지에서:
1. 좌측 상단의 글로벌 네비게이션 아이콘을 클릭해 네비게이션 사이드바를 엽니다.
1. 네비게이션의 **Projects** 섹션에서 **View all**을 클릭해 프로젝트 Overview 페이지로 이동합니다.
1. **Create new project**를 클릭합니다.
1. **Team** 항목에 이 프로젝트를 소유할 팀 이름을 지정합니다.
1. **Name** 필드에 프로젝트 이름을 입력합니다.
1. **Project visibility**를 설정합니다(기본값: **Team**).
1. **Description**을 추가로 입력할 수 있습니다.
1. **Create project**를 클릭합니다.

팀 랜딩 페이지에서:
1. 좌측 상단의 글로벌 네비게이션 아이콘을 클릭해 네비게이션 사이드바를 엽니다.
1. **Teams** 섹션에서 원하는 팀명을 클릭해 해당 팀 랜딩페이지로 이동합니다.
1. 랜딩페이지에서 **Create new project**를 클릭합니다.
1. **Team** 항목은 현재 페이지의 팀으로 자동 설정됩니다. 필요하면 변경할 수 있습니다.
1. **Name** 필드에 프로젝트 이름을 입력합니다.
1. **Project visibility**를 설정합니다(기본값: **Team**).
1. **Description**을 추가로 입력할 수 있습니다.
1. **Create project** 버튼을 클릭합니다.
   {{% /tab %}}
   {{% tab header="Python SDK" %}}
코드로 자동 생성하려면, `wandb.init()` 호출 시 `project` 파라미터를 지정하세요. 해당 프로젝트가 아직 없으면 자동 생성되고, 지정한 entity 소유가 됩니다. 예:

```python
import wandb
# 프로젝트와 entity를 지정하여 run을 시작합니다.
with wandb.init(entity="<entity>", project="<project_name>") as run:
    run.log({"accuracy": .95})
```

API 세부 내용은 [`wandb.init()` API 문서]({{< relref path="/ref/python/sdk/functions/init/#examples" lang="ko" >}})에서 확인하세요.
   {{% /tab %}}  
{{< /tabpane >}}

## 프로젝트에 별표 표시하기

프로젝트에 별을 추가해 중요 프로젝트로 표시할 수 있습니다. 별을 단 프로젝트는 본인과 팀의 조직 홈 상단의 **Starred projects** 영역에 노출됩니다.

예를 들어 아래 이미지는 `zoo_experiment`와 `registry_demo` 프로젝트 두 개가 중요 프로젝트로 별 처리되어 있어, 조직 홈페이지 상단에 표시되고 있습니다.
{{< img src="/images/track/star-projects.png" alt="Starred projects section" >}}


프로젝트에 중요 표시하는 방법은 두 가지입니다: 프로젝트의 overview 탭에서, 또는 팀 프로필 페이지에서 설정할 수 있습니다.

{{< tabpane text=true >}}
    {{% tab header="Project overview" %}}
1. W&B App에서 해당 프로젝트 `https://wandb.ai/<team>/<project-name>`로 이동합니다.
2. 프로젝트 사이드바에서 **Overview** 탭을 선택합니다.
3. 우측 상단의 **Edit** 버튼 옆에 있는 별(★) 아이콘을 클릭하세요.

{{< img src="/images/track/star-project-overview-tab.png" alt="Star project from overview" >}}    
    {{% /tab %}}
    {{% tab header="Team profile" %}}
1. 팀 프로필 페이지 `https://wandb.ai/<team>/projects`로 이동합니다.
2. **Projects** 탭을 선택합니다.
3. 별을 표시하려는 프로젝트 옆에서 마우스를 올리면 별 아이콘이 나타납니다. 해당 아이콘을 클릭하세요.

예를 들어 아래 이미지는 "Compare_Zoo_Models" 프로젝트 옆에 별 아이콘이 표시된 모습입니다.
{{< img src="/images/track/star-project-team-profile-page.png" alt="Star project from team page" >}}    
    {{% /tab %}}
{{< /tabpane >}}

해당 조직 페이지 최상단에서 본인의 프로젝트가 노출되는지 확인하려면 앱 좌측 상단의 조직 이름을 클릭해보세요.


## 프로젝트 삭제하기

overview 탭 우측의 점 세 개 아이콘을 클릭하면 프로젝트를 삭제할 수 있습니다.

{{< img src="/images/app_ui/howto_delete_project.gif" alt="Delete project workflow" >}}

프로젝트에 run 등이 없다면, 우측 상단 드롭다운 메뉴에서 **Delete project**만으로 삭제할 수 있습니다.

{{< img src="/images/app_ui/howto_delete_project_2.png" alt="Delete empty project" >}}



## 프로젝트에 노트 추가하기

프로젝트 별 노트는 Overview의 설명 또는 워크스페이스 내 마크다운 패널 형태로 작성할 수 있습니다.

### 프로젝트에 설명(Overview) 추가하기

Overview 탭에 입력한 설명은 내 프로필의 **Overview** 탭에 표시됩니다.

1. W&B 프로젝트로 이동합니다.
2. 프로젝트 사이드바에서 **Overview** 탭을 선택합니다.
3. 우측 상단 **Edit** 버튼을 클릭합니다.
4. **Description** 필드에 노트를 입력합니다.
5. **Save** 버튼을 클릭합니다.

{{% alert title="여러 run을 비교하는 설명 노트는 리포트로 작성해보세요" %}}
여러 run 결과나 플롯, 마크다운을 나란히 정리하고 싶다면 W&B Report를 이용하세요. 섹션별로 서로 다른 run을 보여주거나, 실험 과정을 스토리로 안내할 수 있습니다.
{{% /alert %}}


### Run 워크스페이스에 노트 추가하기

1. W&B 프로젝트로 이동합니다.
2. 프로젝트 사이드바에서 **Workspace** 탭을 선택합니다.
3. 우측 상단 **Add panels** 버튼을 클릭합니다.
4. 나타나는 모달창에서 **TEXT AND CODE** 드롭다운을 선택합니다.
5. **Markdown**을 선택합니다.
6. 워크스페이스에 생성된 마크다운 패널에 노트를 입력합니다.