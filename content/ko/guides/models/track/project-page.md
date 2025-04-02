---
title: Projects
description: 모델의 버전을 비교하고, 임시 워크스페이스에서 결과를 탐색하고, 발견한 내용을 리포트로 내보내 노<br>트와 시각화를 저장하세요.
menu:
  default:
    identifier: ko-guides-models-track-project-page
    parent: experiments
weight: 3
---

*프로젝트*는 결과를 시각화하고, 실험을 비교하고, 아티팩트를 보고 다운로드하고, 자동화를 생성하는 등 다양한 작업을 수행할 수 있는 중앙 위치입니다.

{{% alert %}}
각 프로젝트에는 누가 엑세스할 수 있는지 결정하는 공개 설정이 있습니다. 프로젝트에 엑세스할 수 있는 사용자에 대한 자세한 내용은 [프로젝트 공개 설정]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ko" >}})을 참조하세요.
{{% /alert %}}

각 프로젝트에는 사이드바에서 엑세스할 수 있는 다음이 포함되어 있습니다.

* [**Overview**]({{< relref path="project-page.md#overview-tab" lang="ko" >}}): 프로젝트의 스냅샷
* [**Workspace**]({{< relref path="project-page.md#workspace-tab" lang="ko" >}}): 개인 시각화 샌드박스
* [**Runs**]({{< relref path="#runs-tab" lang="ko" >}}): 프로젝트의 모든 run을 나열하는 테이블
* **Automations**: 프로젝트에서 구성된 자동화
* [**Sweeps**]({{< relref path="project-page.md#sweeps-tab" lang="ko" >}}): 자동화된 탐색 및 최적화
* [**Reports**]({{< relref path="project-page.md#reports-tab" lang="ko" >}}): 노트, run 및 그래프의 저장된 스냅샷
* [**Artifacts**]({{< relref path="#artifacts-tab" lang="ko" >}}): 모든 run과 해당 run과 연결된 아티팩트 포함

## Overview 탭

* **Project name**: 프로젝트 이름입니다. W&B는 프로젝트 필드에 제공한 이름으로 run을 초기화할 때 프로젝트를 생성합니다. 오른쪽 상단 모서리에 있는 **편집** 버튼을 선택하여 언제든지 프로젝트 이름을 변경할 수 있습니다.
* **Description**: 프로젝트에 대한 설명입니다.
* **Project visibility**: 프로젝트의 공개 설정입니다. 누가 엑세스할 수 있는지 결정하는 공개 설정입니다. 자세한 내용은 [프로젝트 공개 설정]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ko" >}})을 참조하세요.
* **Last active**: 이 프로젝트에 마지막으로 데이터가 기록된 타임스탬프
* **Owner**: 이 프로젝트를 소유한 엔티티
* **Contributors**: 이 프로젝트에 기여하는 Users 수
* **Total runs**: 이 프로젝트의 총 Runs 수
* **Total compute**: 프로젝트의 모든 run 시간을 합산하여 이 총계를 얻습니다.
* **Undelete runs**: 드롭다운 메뉴를 클릭하고 "Undelete all runs"를 클릭하여 프로젝트에서 삭제된 Runs을 복구합니다.
* **Delete project**: 오른쪽 상단 모서리에 있는 점 메뉴를 클릭하여 프로젝트를 삭제합니다.

[라이브 예제 보기](https://app.wandb.ai/example-team/sweep-demo/overview)

{{< img src="/images/track/overview_tab_image.png" alt="" >}}

## Workspace 탭

프로젝트의 *workspace*는 실험을 비교할 수 있는 개인 샌드박스를 제공합니다. 프로젝트를 사용하여 다양한 아키텍처, 하이퍼파라미터, 데이터셋, 전처리 등으로 동일한 문제에 대해 작업하면서 비교할 수 있는 Models를 구성합니다.

**Runs Sidebar**: 프로젝트의 모든 Runs 목록입니다.

* **Dot menu**: 사이드바에서 행 위로 마우스를 가져가면 왼쪽에 메뉴가 나타납니다. 이 메뉴를 사용하여 run 이름을 바꾸거나, run을 삭제하거나, 활성 run을 중지합니다.
* **Visibility icon**: 눈을 클릭하여 그래프에서 Runs을 켜고 끕니다.
* **Color**: run 색상을 다른 사전 설정 색상 또는 사용자 지정 색상으로 변경합니다.
* **Search**: 이름으로 Runs을 검색합니다. 이렇게 하면 플롯에서 보이는 Runs도 필터링됩니다.
* **Filter**: 사이드바 필터를 사용하여 보이는 Runs 집합을 좁힙니다.
* **Group**: 아키텍처별로 Runs을 동적으로 그룹화할 구성 열을 선택합니다. 그룹화하면 플롯에 평균 값을 따라 선이 표시되고 그래프에서 점의 분산에 대한 음영 영역이 표시됩니다.
* **Sort**: 가장 낮은 손실 또는 가장 높은 정확도를 가진 Runs과 같이 Runs을 정렬할 값을 선택합니다. 정렬은 그래프에 표시되는 Runs에 영향을 미칩니다.
* **Expand button**: 사이드바를 전체 테이블로 확장합니다.
* **Run count**: 상단의 괄호 안의 숫자는 프로젝트의 총 Runs 수입니다. (N visualized) 숫자는 눈이 켜져 있고 각 플롯에서 시각화할 수 있는 Runs 수입니다. 아래 예에서 그래프는 183개의 Runs 중 처음 10개만 보여줍니다. 보이는 Runs의 최대 수를 늘리려면 그래프를 편집합니다.

[Runs 탭](#runs-tab)에서 열을 고정, 숨기거나 순서를 변경하면 Runs 사이드바에 이러한 사용자 지정이 반영됩니다.

**Panels layout**: 이 스크래치 공간을 사용하여 결과를 탐색하고, 차트를 추가 및 제거하고, 다양한 메트릭을 기반으로 Models 버전을 비교합니다.

[라이브 예제 보기](https://app.wandb.ai/example-team/sweep-demo)

{{< img src="/images/app_ui/workspace_tab_example.png" alt="" >}}

### 패널 섹션 추가

섹션 드롭다운 메뉴를 클릭하고 "섹션 추가"를 클릭하여 패널에 대한 새 섹션을 만듭니다. 섹션 이름을 바꾸고, 드래그하여 재구성하고, 섹션을 확장 및 축소할 수 있습니다.

각 섹션에는 오른쪽 상단 모서리에 다음과 같은 옵션이 있습니다.

* **Switch to custom layout**: 사용자 지정 레이아웃을 사용하면 패널 크기를 개별적으로 조정할 수 있습니다.
* **Switch to standard layout**: 표준 레이아웃을 사용하면 섹션의 모든 패널 크기를 한 번에 조정할 수 있으며 페이지 매김을 제공합니다.
* **Add section**: 드롭다운 메뉴에서 위 또는 아래에 섹션을 추가하거나 페이지 하단의 버튼을 클릭하여 새 섹션을 추가합니다.
* **Rename section**: 섹션 제목을 변경합니다.
* **Export section to report**: 이 패널 섹션을 새 Report에 저장합니다.
* **Delete section**: 전체 섹션과 모든 차트를 제거합니다. 페이지 하단의 워크스페이스 바에서 실행 취소 버튼으로 실행 취소할 수 있습니다.
* **Add panel**: 더하기 버튼을 클릭하여 섹션에 패널을 추가합니다.

{{< img src="/images/app_ui/add-section.gif" alt="" >}}

### 섹션 간에 패널 이동

패널을 드래그 앤 드롭하여 섹션으로 재정렬하고 구성합니다. 패널 오른쪽 상단 모서리에 있는 "이동" 버튼을 클릭하여 패널을 이동할 섹션을 선택할 수도 있습니다.

{{< img src="/images/app_ui/move-panel.gif" alt="" >}}

### 패널 크기 조정

* **Standard layout**: 모든 패널은 동일한 크기를 유지하고 패널 페이지가 있습니다. 오른쪽 하단 모서리를 클릭하고 드래그하여 패널 크기를 조정할 수 있습니다. 섹션의 오른쪽 하단 모서리를 클릭하고 드래그하여 섹션 크기를 조정합니다.
* **Custom layout**: 모든 패널의 크기는 개별적으로 조정되며 페이지가 없습니다.

{{< img src="/images/app_ui/resize-panel.gif" alt="" >}}

### 메트릭 검색

워크스페이스의 검색 상자를 사용하여 패널을 필터링합니다. 이 검색은 기본적으로 시각화된 메트릭의 이름인 패널 제목과 일치합니다.

{{< img src="/images/app_ui/search_in_the_workspace.png" alt="" >}}

## Runs 탭

Runs 탭을 사용하여 Runs을 필터링, 그룹화 및 정렬합니다.

{{< img src="/images/runs/run-table-example.png" alt="" >}}

다음 탭은 Runs 탭에서 수행할 수 있는 몇 가지 일반적인 작업을 보여줍니다.

{{< tabpane text=true >}}
   {{% tab header="Customize columns" %}}
Runs 탭은 프로젝트의 Runs에 대한 세부 정보를 보여줍니다. 기본적으로 많은 수의 열을 보여줍니다.

- 보이는 모든 열을 보려면 페이지를 가로로 스크롤합니다.
- 열 순서를 변경하려면 열을 왼쪽 또는 오른쪽으로 드래그합니다.
- 열을 고정하려면 열 이름 위로 마우스를 가져간 다음 나타나는 작업 메뉴 `...`을 클릭한 다음 **열 고정**을 클릭합니다. 고정된 열은 **이름** 열 뒤에 페이지 왼쪽에 가깝게 나타납니다. 고정된 열을 고정 해제하려면 **열 고정 해제**를 선택합니다.
- 열을 숨기려면 열 이름 위로 마우스를 가져간 다음 나타나는 작업 메뉴 `...`을 클릭한 다음 **열 숨기기**를 클릭합니다. 현재 숨겨진 모든 열을 보려면 **열**을 클릭합니다.
  - 숨겨진 열의 이름을 클릭하여 숨김 해제합니다.
  - 보이는 열의 이름을 클릭하여 숨깁니다.
  - 보이는 열 옆에 있는 핀 아이콘을 클릭하여 고정합니다.

Runs 탭을 사용자 지정하면 사용자 지정은 [Workspace 탭]({{< relref path="#workspace-tab" lang="ko" >}})의 **Runs** 선택기에도 반영됩니다.
   {{% /tab %}}

   {{% tab header="Sort" %}}
지정된 열의 값을 기준으로 테이블의 모든 행을 정렬합니다.

1. 열 제목 위로 마우스를 가져갑니다. 케밥 메뉴(세 개의 세로 문서)가 나타납니다.
2. 케밥 메뉴(세 개의 세로 점)에서 선택합니다.
3. 행을 오름차순 또는 내림차순으로 정렬하려면 각각 **오름차순 정렬** 또는 **내림차순 정렬**을 선택합니다.

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="모델이 가장 자신 있게 '0'으로 추정한 숫자를 참조하세요." >}}

위의 이미지는 `val_acc`라는 테이블 열에 대한 정렬 옵션을 보는 방법을 보여줍니다.
   {{% /tab %}}
   {{% tab header="Filter" %}}
대시보드 왼쪽 상단에 있는 **필터** 버튼을 사용하여 표현식으로 모든 행을 필터링합니다.

{{< img src="/images/data_vis/filter.png" alt="모델이 잘못 인식하는 예만 참조하세요." >}}

**필터 추가**를 선택하여 행에 하나 이상의 필터를 추가합니다. 세 개의 드롭다운 메뉴가 나타납니다. 왼쪽에서 오른쪽으로 필터 유형은 열 이름, 연산자 및 값을 기반으로 합니다.

|                   | 열 이름 | 이항 관계    | 값       |
| -----------       | ----------- | ----------- | ----------- |
| 허용된 값   | 문자열       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | 정수, 부동 소수점, 문자열, 타임스탬프, null |

표현식 편집기는 열 이름과 논리적 술어 구조에 대한 자동 완성을 사용하여 각 용어에 대한 옵션 목록을 보여줍니다. "and" 또는 "or"(때로는 괄호)를 사용하여 여러 논리적 술어를 하나의 표현식으로 연결할 수 있습니다.

{{< img src="/images/data_vis/filter_example.png" alt="" >}}
위의 이미지는 `val_loss` 열을 기반으로 하는 필터를 보여줍니다. 필터는 유효성 검사 손실이 1 이하인 Runs을 보여줍니다.
   {{% /tab %}}
   {{% tab header="Group" %}}
열 머리글에서 **그룹화 기준** 버튼을 사용하여 특정 열의 값을 기준으로 모든 행을 그룹화합니다.

{{< img src="/images/data_vis/group.png" alt="진실 분포는 작은 오류를 보여줍니다. 8과 2는 7과 9로, 9는 2로 착각합니다." >}}

기본적으로 이렇게 하면 다른 숫자 열이 해당 그룹 전체의 해당 열에 대한 값 분포를 보여주는 히스토그램으로 바뀝니다. 그룹화는 데이터에서 상위 수준 패턴을 이해하는 데 유용합니다.
   {{% /tab %}}
{{< /tabpane >}}

## Reports 탭

결과의 모든 스냅샷을 한 곳에서 보고 팀과 발견한 내용을 공유하세요.

{{< img src="/images/app_ui/reports-tab.png" alt="" >}}

## Sweeps 탭

프로젝트에서 새 [sweep]({{< relref path="/guides/models/sweeps/" lang="ko" >}})을 시작합니다.

{{< img src="/images/app_ui/sweeps-tab.png" alt="" >}}

## Artifacts 탭

트레이닝 데이터셋 및 [파인튜닝된 Models]({{< relref path="/guides/core/registry/" lang="ko" >}})에서 [메트릭 및 미디어 테이블]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ko" >}})에 이르기까지 프로젝트와 연결된 모든 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 봅니다.

### Overview 패널

{{< img src="/images/app_ui/overview_panel.png" alt="" >}}

Overview 패널에서는 아티팩트 이름과 버전, 변경 사항을 감지하고 중복을 방지하는 데 사용되는 해시 다이제스트, 생성 날짜, 에일리어스를 포함하여 아티팩트에 대한 다양한 고급 정보를 찾을 수 있습니다. 여기서 에일리어스를 추가하거나 제거하고 버전과 아티팩트 전체에 대한 메모를 작성할 수 있습니다.

### Metadata 패널

{{< img src="/images/app_ui/metadata_panel.png" alt="" >}}

Metadata 패널은 아티팩트가 구성될 때 제공되는 아티팩트의 메타데이터에 대한 엑세스를 제공합니다. 이 메타데이터에는 아티팩트를 재구성하는 데 필요한 구성 인수, 더 많은 정보를 찾을 수 있는 URL 또는 아티팩트를 기록한 run 중에 생성된 메트릭이 포함될 수 있습니다. 또한 아티팩트를 생성한 run에 대한 구성과 아티팩트를 로깅할 당시의 기록 메트릭을 볼 수 있습니다.

### Usage 패널

{{< img src="/images/app_ui/usage_panel.png" alt="" >}}

Usage 패널은 예를 들어 로컬 머신에서 웹 앱 외부에서 사용할 수 있도록 아티팩트를 다운로드하기 위한 코드 조각을 제공합니다. 이 섹션은 또한 아티팩트를 출력하는 run과 아티팩트를 입력으로 사용하는 모든 Runs을 나타내고 링크합니다.

### Files 패널

{{< img src="/images/app_ui/files_panel.png" alt="" >}}

Files 패널은 아티팩트와 연결된 파일 및 폴더를 나열합니다. W&B는 run에 대한 특정 파일을 자동으로 업로드합니다. 예를 들어 `requirements.txt`는 run에서 사용된 각 라이브러리의 버전을 보여주고 `wandb-metadata.json` 및 `wandb-summary.json`에는 run에 대한 정보가 포함되어 있습니다. 다른 파일은 run의 구성에 따라 아티팩트 또는 미디어와 같이 업로드될 수 있습니다. 이 파일 트리를 탐색하고 W&B 웹 앱에서 직접 내용을 볼 수 있습니다.

아티팩트와 연결된 [테이블]({{< relref path="/guides/models/tables//tables-walkthrough.md" lang="ko" >}})은 특히 풍부하고 대화형입니다. Artifacts와 함께 테이블을 사용하는 방법에 대해 자세히 알아보려면 [여기]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ko" >}})를 참조하세요.

{{< img src="/images/app_ui/files_panel_table.png" alt="" >}}

### Lineage 패널

{{< img src="/images/app_ui/lineage_panel.png" alt="" >}}

Lineage 패널은 프로젝트와 연결된 모든 아티팩트와 서로 연결하는 Runs에 대한 뷰를 제공합니다. run 유형을 블록으로, 아티팩트를 원으로 표시하고 화살표를 사용하여 지정된 유형의 run이 지정된 유형의 아티팩트를 소비하거나 생성하는 시기를 나타냅니다. 왼쪽 열에서 선택한 특정 아티팩트의 유형이 강조 표시됩니다.

개별 아티팩트 버전과 연결하는 특정 Runs을 모두 보려면 Explode 토글을 클릭합니다.

### Action History Audit 탭

{{< img src="/images/app_ui/action_history_audit_tab_1.png" alt="" >}}

{{< img src="/images/app_ui/action_history_audit_tab_2.png" alt="" >}}

작업 기록 감사 탭은 리소스의 전체 진화를 감사할 수 있도록 컬렉션에 대한 모든 에일리어스 작업과 멤버십 변경 사항을 보여줍니다.

### Versions 탭

{{< img src="/images/app_ui/versions_tab.png" alt="" >}}

Versions 탭은 아티팩트의 모든 버전과 버전을 로깅할 당시의 Run History의 각 숫자 값에 대한 열을 보여줍니다. 이를 통해 성능을 비교하고 관심 있는 버전을 빠르게 식별할 수 있습니다.

## 프로젝트에 별표 표시

프로젝트에 별표를 추가하여 해당 프로젝트를 중요하다고 표시합니다. 사용자와 팀이 별표로 중요하다고 표시한 프로젝트는 조직의 홈페이지 상단에 나타납니다.

예를 들어, 다음 이미지는 중요하다고 표시된 두 개의 프로젝트인 `zoo_experiment`와 `registry_demo`를 보여줍니다. 두 프로젝트 모두 **Starred projects** 섹션 내에서 조직의 홈페이지 상단에 나타납니다.
{{< img src="/images/track/star-projects.png" alt="" >}}

프로젝트를 중요하다고 표시하는 방법에는 프로젝트의 Overview 탭 내에서 또는 팀의 프로필 페이지 내에서 두 가지가 있습니다.

{{< tabpane text=true >}}
    {{% tab header="Project overview" %}}
1. W&B 앱의 `https://wandb.ai/<team>/<project-name>`에서 W&B 프로젝트로 이동합니다.
2. 프로젝트 사이드바에서 **Overview** 탭을 선택합니다.
3. 오른쪽 상단 모서리에 있는 **편집** 버튼 옆에 있는 별표 아이콘을 선택합니다.

{{< img src="/images/track/star-project-overview-tab.png" alt="" >}}
    {{% /tab %}}
    {{% tab header="Team profile" %}}
1. `https://wandb.ai/<team>/projects`에서 팀의 프로필 페이지로 이동합니다.
2. **Projects** 탭을 선택합니다.
3. 별표를 표시할 프로젝트 옆으로 마우스를 가져갑니다. 나타나는 별표 아이콘을 클릭합니다.

예를 들어, 다음 이미지는 "Compare_Zoo_Models" 프로젝트 옆에 있는 별표 아이콘을 보여줍니다.
{{< img src="/images/track/star-project-team-profile-page.png" alt="" >}}
    {{% /tab %}}
{{< /tabpane >}}

앱의 왼쪽 상단 모서리에 있는 조직 이름을 클릭하여 프로젝트가 조직의 랜딩 페이지에 나타나는지 확인합니다.

## 프로젝트 삭제

Overview 탭의 오른쪽에 있는 세 개의 점을 클릭하여 프로젝트를 삭제할 수 있습니다.

{{< img src="/images/app_ui/howto_delete_project.gif" alt="" >}}

프로젝트가 비어 있으면 오른쪽 상단의 드롭다운 메뉴를 클릭하고 **프로젝트 삭제**를 선택하여 삭제할 수 있습니다.

{{< img src="/images/app_ui/howto_delete_project_2.png" alt="" >}}

## 프로젝트에 노트 추가

설명 개요 또는 Workspace 내의 마크다운 패널로 프로젝트에 노트를 추가합니다.

### 프로젝트에 설명 개요 추가

페이지에 추가하는 설명은 프로필의 **Overview** 탭에 나타납니다.

1. W&B 프로젝트로 이동합니다.
2. 프로젝트 사이드바에서 **Overview** 탭을 선택합니다.
3. 오른쪽 상단 모서리에서 편집을 선택합니다.
4. **Description** 필드에 노트를 추가합니다.
5. **Save** 버튼을 선택합니다.

{{% alert title="Runs을 비교하는 설명 노트를 만들려면 리포트를 만드세요." %}}
W&B Report를 만들어 플롯과 마크다운을 나란히 추가할 수도 있습니다. 다양한 섹션을 사용하여 다양한 Runs을 보여주고 작업한 내용에 대한 스토리를 전달합니다.
{{% /alert %}}

### Run Workspace에 노트 추가

1. W&B 프로젝트로 이동합니다.
2. 프로젝트 사이드바에서 **Workspace** 탭을 선택합니다.
3. 오른쪽 상단 모서리에서 **패널 추가** 버튼을 선택합니다.
4. 나타나는 모달에서 **TEXT AND CODE** 드롭다운을 선택합니다.
5. **Markdown**을 선택합니다.
6. Workspace에 나타나는 마크다운 패널에 노트를 추가합니다.
