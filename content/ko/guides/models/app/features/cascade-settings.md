---
title: 워크스페이스, 섹션, 패널 설정 관리
menu:
  default:
    identifier: ko-guides-models-app-features-cascade-settings
    parent: w-b-app-ui-reference
url: guides/app/features/cascade-settings
---

특정 워크스페이스 페이지 내에는 세 가지 다른 설정 수준이 있습니다: 워크스페이스, 섹션, 그리고 패널입니다. [워크스페이스 설정]({{< relref path="#workspace-settings" lang="ko" >}})은 워크스페이스 전체에 적용됩니다. [섹션 설정]({{< relref path="#section-settings" lang="ko" >}})은 해당 섹션 내 모든 패널에 적용됩니다. [패널 설정]({{< relref path="#panel-settings" lang="ko" >}})은 개별 패널에 적용됩니다.

## 워크스페이스 설정

워크스페이스 설정은 모든 섹션 및 해당 섹션 내의 모든 패널에 적용됩니다. 워크스페이스에서 두 가지 유형의 설정을 변경할 수 있습니다: [워크스페이스 레이아웃]({{< relref path="#workspace-layout-options" lang="ko" >}})과 [라인 플롯]({{< relref path="#line-plots-options" lang="ko" >}})입니다. **워크스페이스 레이아웃**은 워크스페이스의 구조를 결정하며, **라인 플롯** 설정은 워크스페이스에서 사용되는 라인 플롯의 기본 설정을 제어합니다.

워크스페이스의 전체 구조에 적용되는 설정을 편집하려면:

1. 프로젝트 워크스페이스로 이동합니다.
2. **New report** 버튼 옆에 있는 톱니바퀴 아이콘을 클릭하여 워크스페이스 설정을 확인합니다.
3. **Workspace layout**을 선택해 워크스페이스 레이아웃을 변경하거나, **Line plots**를 선택해 라인 플롯의 기본 설정을 구성합니다.
{{< img src="/images/app_ui/workspace_settings.png" alt="Workspace settings gear icon" >}}

{{% alert %}}
워크스페이스를 원하는 대로 커스터마이즈한 후에는 _워크스페이스 템플릿_을 이용해 동일한 설정으로 새로운 워크스페이스를 빠르게 만들 수 있습니다. 자세한 내용은 [워크스페이스 템플릿]({{< relref path="/guides/models/track/workspaces.md#workspace-templates" lang="ko" >}})을 참고하세요.
{{% /alert %}}

### 워크스페이스 레이아웃 옵션

워크스페이스 레이아웃을 구성하여 워크스페이스의 전체 구조를 정의할 수 있습니다. 여기에는 섹션 구분 방식과 패널 배치가 포함됩니다.

{{< img src="/images/app_ui/workspace_layout_settings.png" alt="Workspace layout options" >}}

워크스페이스 레이아웃 옵션 페이지에서는 패널이 자동 생성되는지 수동으로 생성되는지 확인할 수 있습니다. 워크스페이스의 패널 생성 방식을 변경하려면 [Panels]({{< relref path="panels/" lang="ko" >}})을 참고하세요.

아래 표는 각 워크스페이스 레이아웃 옵션에 대해 설명합니다.

| 워크스페이스 설정 | 설명 |
| ----- | ----- |
| **검색 시 비어있는 섹션 숨기기** | 패널이 없는 섹션을 패널 검색 시 숨겨줍니다. |
| **패널을 알파벳 순으로 정렬** | 워크스페이스 내의 패널을 알파벳 순으로 정렬합니다. |
| **섹션 재구성** | 기존 섹션 및 패널을 모두 제거하고 새로운 섹션 이름으로 다시 구성합니다. 새로 추가된 섹션을 앞 혹은 뒤의 접두사로 그룹화합니다. |

{{% alert %}}
W&B에서는 접두사를 마지막이 아닌 처음 기준으로 그룹핑하여 섹션을 구성하는 것이 더 효율적이라고 권장합니다. 접두사를 처음으로 그룹핑하면 섹션 수가 줄어들어 성능이 향상될 수 있습니다.
{{% /alert %}}

### 라인 플롯 옵션
워크스페이스에서 **라인 플롯** 워크스페이스 설정을 수정하여 라인 플롯에 대한 전역 기본값 및 사용자 지정 규칙을 설정할 수 있습니다.

{{< img src="/images/app_ui/workspace_settings_line_plots.png" alt="Line plot settings" >}}

**라인 플롯** 설정 내에서 두 가지 주요 설정을 편집할 수 있습니다: **Data**와 **Display preferences**입니다. **Data** 탭에서는 다음과 같은 설정이 포함되어 있습니다:

| 라인 플롯 설정 | 설명 |
| ----- | ----- |
| **X축** | 라인 플롯에서 x축의 스케일을 설정합니다. 기본값은 **Step**입니다. 사용 가능한 x축 옵션 목록은 아래 표를 참고하세요. |
| **범위** | x축에 표시될 최소 및 최대 값을 지정합니다. |
| **스무딩** | 라인 플롯의 스무딩 값을 변경합니다. 자세한 내용은 [스무딩된 라인 플롯]({{< relref path="/guides/models/app/features/panels/line-plot/smoothing.md" lang="ko" >}})에서 확인하세요. |
| **이상치** | 이상치를 제외하여 기본 플롯의 최소값과 최대값을 다시 계산합니다. |
| **포인트 집계 메소드** | 데이터 시각화의 정확성과 성능을 높입니다. 자세한 내용은 [포인트 집계]({{< relref path="/guides/models/app/features/panels/line-plot/sampling.md" lang="ko" >}})를 참고하세요. |
| **최대 runs 또는 그룹 수** | 라인 플롯에 표시될 runs 또는 그룹의 수를 제한합니다. |

**Step** 외에도 x축에는 다음과 같은 옵션이 있습니다:

| X축 옵션 | 설명 |
| ------------- | ----------- |
| **Relative Time (Wall)**| 프로세스 시작 이후 경과 시간(타임스탬프). 예를 들어, run을 시작한 후 다음 날 재개하면, 24시간으로 기록됩니다. |
| **Relative Time (Process)** | 실행 중인 프로세스 내에서의 경과 시간(타임스탬프). 예를 들어, run을 시작하여 10초간 실행하고, 다음 날 다시 실행할 경우 10초로 기록됩니다. |
| **Wall Time** | 그래프에서 첫 번째 run 시작 이후 경과된 분 단위 시간입니다. |
| **Step** | `wandb.Run.log()`를 호출할 때마다 증가합니다. |

{{% alert %}}
개별 라인 플롯 수정 방법은 [라인 패널 설정 편집]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ko" >}})를 참고하세요.
{{% /alert %}}

**Display preferences** 탭에서는 아래와 같은 설정을 전환할 수 있습니다:

| 표시 환경설정 | 설명 |
| ----- | ----- |
| **모든 패널에서 범례 제거** | 패널의 범례(legend)를 제거합니다. |
| **툴팁에서 컬러 run 이름 표시** | 툴팁 내에서 run 이름을 컬러 텍스트로 표시합니다. |
| **보조 차트 툴팁에 강조된 run만 표시** | 차트 툴팁에 강조된 run만 표시합니다. |
| **툴팁에 표시되는 run 수** | 툴팁에 표시되는 run의 수를 지정합니다. |
| **주요 차트 툴팁에서 전체 run 이름 표시**| 차트 툴팁에 run의 전체 이름을 표시합니다. |



## 섹션 설정

섹션 설정은 해당 섹션 내의 모든 패널에 적용됩니다. 워크스페이스 섹션 내에서 패널 정렬, 패널 순서 변경, 섹션 이름 변경 등의 작업을 할 수 있습니다.

섹션의 오른쪽 상단에 있는 가로 점 세 개 (**...**)를 클릭해 섹션 설정을 수정하세요.

{{< img src="/images/app_ui/section_settings.png" alt="Section settings menu" >}}

드롭다운 메뉴에서 섹션 전체에 적용되는 다음 설정을 변경할 수 있습니다:

| 섹션 설정 | 설명 |
| ----- | ----- |
| **섹션 이름 변경** | 섹션 이름을 변경합니다. |
| **패널을 A-Z로 정렬** | 섹션 내 패널을 알파벳 순으로 정렬합니다. |
| **패널 순서 재배치** | 섹션 내의 패널을 선택 및 드래그하여 원하는 순서로 배치할 수 있습니다. |

아래 애니메이션은 섹션 내에서 패널을 재배치하는 방법을 보여줍니다:

{{< img src="/images/app_ui/rearrange_panels.gif" alt="Rearranging panels" >}}

{{% alert %}}
위 표에 설명된 설정 외에도 **Add section below**, **Add section above**, **Delete section**, **Add section to report**와 같은 섹션의 표시 방식을 워크스페이스 내에서 편집할 수 있습니다.
{{% /alert %}}

## 패널 설정

개별 패널의 설정을 커스터마이즈하여 한 플롯에서 여러 라인을 비교하거나, 커스텀 축을 계산하거나, 라벨 이름을 변경하는 등 다양한 작업을 할 수 있습니다. 패널 설정을 변경하려면:

1. 수정하려는 패널에 마우스를 가져다 놓습니다.
2. 나타나는 연필 아이콘을 선택합니다.
{{< img src="/images/app_ui/panel_settings.png" alt="Panel edit icon" >}}
3. 표시되는 모달 내에서, 패널 데이터, 표시 환경설정 등 여러 설정 항목을 편집할 수 있습니다.
{{< img src="/images/app_ui/panel_settings_modal.png" alt="Panel settings modal" >}}

패널에 적용할 수 있는 전체 설정 목록은 [라인 패널 설정 편집]({{< relref path="/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" lang="ko" >}})에서 확인할 수 있습니다.