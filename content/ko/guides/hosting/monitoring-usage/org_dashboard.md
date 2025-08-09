---
title: 조직 활동 보기
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-org_dashboard
    parent: monitoring-and-usage
---

이 페이지에서는 W&B 조직 내에서 활동을 확인하는 다양한 방법을 안내합니다.

## 사용자 상태 및 활동 보기

{{< tabpane text=true >}}
{{% tab header="전용 / 자체 관리형" value="dedicated" %}}
1. **Organization Dashboard**에 엑세스하려면 `https://<org-name>.io/org/dashboard/`로 이동하세요. `<org-name>`을 조직 이름으로 바꿔 입력합니다. 기본적으로 **Users** 탭이 열리며, 모든 사용자와 각 사용자에 대한 데이터가 표시됩니다.
1. 사용자 상태로 리스트를 정렬하려면 **Last Active** 열 라벨을 클릭하세요. 각 사용자의 상태는 다음 중 하나입니다:

    * **Invite pending**: 관리자가 초대를 보냈으나 사용자가 아직 수락하지 않은 상태입니다.
    * **Active**: 사용자가 초대를 수락하고 계정을 생성한 상태입니다.
    * **-**: 이전에 활동적이었으나 최근 6개월 동안 활동하지 않은 사용자입니다.
    * **Deactivated**: 관리자가 해당 사용자의 엑세스를 회수한 상태입니다.
1. 특정 사용자의 최근 활동 세부 정보를 보려면 해당 사용자의 **Last Active** 필드 위에 마우스를 올려보세요. 툴팁이 나타나며, 사용자가 추가된 시기와 총 활동 일수를 알려줍니다.

    사용자가 _active_로 간주되는 경우는 다음과 같습니다:
    - W&B에 로그인할 때
    - W&B App의 어떤 페이지든 볼 때
    - run을 로그할 때
    - SDK로 experiment을 추적할 때
    - W&B Server와 어떤 방식으로든 상호작용할 때
{{% /tab %}}

{{% tab header="멀티테넌트 클라우드" value="saas" %}}
1. [**Members** 페이지](https://wandb.ai/account-settings/wandb/members/)로 이동하세요. 이 페이지에서는 모든 사용자와 각 사용자에 대한 데이터를 확인할 수 있습니다.
1. 사용자 상태로 리스트를 정렬하려면 **Last Active** 열 라벨을 클릭하세요. 각 사용자의 상태는 다음 중 하나입니다:

    * **Invite pending**: 관리자가 초대를 보냈으나 사용자가 아직 수락하지 않은 상태입니다.
    * **Active**: 사용자가 초대를 수락하고 계정을 생성한 상태입니다.
    * `-`: 이 하이픈은 사용자가 아직 이 조직 내에서 활동한 적이 없음을 의미합니다.

    사용자가 _active_로 간주되는 경우는 조직 단위에서 _2025년 5월 8일 이후_ 감시 가능한 행동을 수행한 경우입니다. 전체 목록은 [Actions]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md#actions" lang="ko" >}})에서 확인할 수 있습니다.
{{% /tab %}}
{{< /tabpane >}}

## 사용자 정보 내보내기

{{< tabpane text=true >}}
{{% tab header="전용 또는 자체 관리형" value="dedicated" %}}
**Users** 탭에서 조직이 W&B를 어떻게 사용하는지에 대한 세부 정보를 CSV 형식으로 내보낼 수 있습니다.

1. **Organization Dashboard**에서 `https://<org-name>.io/org/dashboard/`로 이동하세요. `<org-name>`을 조직 이름으로 바꿔 입력합니다. 기본적으로 **Users** 탭이 열립니다.
1. **Invite new user user** 버튼 옆의 동작 `...` 메뉴를 클릭하세요.
1. **Export as CSV**를 클릭하세요. 다운로드된 CSV 파일에는 각 사용자의 이름과 이메일 어드레스, 최근 활동 시각, 역할 등 조직 내 사용자에 관한 상세 정보가 포함됩니다.
{{% /tab %}}

{{% tab header="멀티테넌트 클라우드" value="saas" %}}
멀티테넌트 클라우드에서는 사용자 내보내기 기능을 제공하지 않습니다.
{{% /tab %}}
{{< /tabpane >}}

## 시간별 활동 보기
이 섹션에서는 시간별 활동의 집계된 뷰를 확인하는 방법을 안내합니다.

{{< tabpane text=true >}}
{{% tab header="전용 또는 자체 관리형" value="dedicated" %}}

**Activity** 탭의 플롯을 사용해 일정 기간 동안 몇 명의 사용자가 활동했는지 집계 뷰를 볼 수 있습니다.

1. **Organization Dashboard**에 엑세스하려면 `https://<org-name>.io/org/dashboard/`로 이동하세요. `<org-name>`을 조직 이름으로 바꿔 입력합니다.
1. **Activity** 탭을 클릭하세요.
1. **Total active users** 플롯은 일정 기간(기본 3개월) 동안 고유하게 활동한 사용자 수를 보여줍니다.
1. **Users active over time** 플롯은 일정 기간(기본 6개월) 동안 활동한 사용자 수의 변동을 보여줍니다. 점 위에 마우스를 올려보면 해당 날짜의 사용자 수를 확인할 수 있습니다.

플롯의 기간을 변경하려면 드롭다운을 사용하세요. 선택 가능한 옵션은 다음과 같습니다:
- 최근 30일
- 최근 3개월
- 최근 6개월
- 최근 12개월
- 전체 기간

{{% /tab %}}
{{% tab header="멀티테넌트 클라우드" value="saas" %}}

**Activity Dashboard**의 플롯을 사용해 시간에 따른 활동의 집계 뷰를 확인할 수 있습니다:

1. 우측 상단의 사용자 프로필 아이콘을 클릭하세요.
1. **Account** 아래에서 **Users**를 클릭하세요.
1. 사용자 목록 위에 있는 Activity Panel을 확인하세요. 여기에는 다음이 표시됩니다:

  - **Active user count** 배지는 일정 기간(기본 3개월) 동안 고유하게 활동한 사용자 수를 보여줍니다. 사용자가 _active_로 간주되는 경우는 조직 단위에서 감시 가능한 행동을 수행했을 때입니다. 전체 목록은 [Actions]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md#actions" lang="ko" >}})에서 확인할 수 있습니다.
  - **Weekly active users** 플롯은 주별로 활동한 사용자 수를 보여줍니다.
  - **Most active user** 리더보드는 해당 기간 동안 활동한 일수를 기준으로 가장 활동적인 상위 10명의 사용자를 나타내며, 각 사용자가 마지막으로 활동한 시점도 보여줍니다.

1. 플롯이 보여주는 기간을 조정하려면 우측 상단의 날짜 선택기를 클릭하세요. 7일, 30일, 90일 중에서 선택할 수 있습니다. 기본값은 30일입니다. 모든 플롯은 동일한 기간을 공유하며 자동으로 업데이트됩니다.

{{% /tab %}}
{{< /tabpane >}}