---
title: View organization dashboard
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-org_dashboard
    parent: monitoring-and-usage
---

{{% alert color="secondary" %}}
Organization dashboard는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 및 [자체 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}})에서만 사용할 수 있습니다.
{{% /alert %}}

## W&B의 조직 사용량 보기
조직 대시보드를 사용하여 조직의 W&B 사용량에 대한 전체적인 보기를 얻을 수 있습니다. 대시보드는 탭별로 구성되어 있습니다.

- **Users**: 이름, 이메일, Teams, 역할 및 마지막 활동을 포함하여 각 user에 대한 세부 정보를 나열합니다.
- **Service accounts**: 서비스 계정에 대한 세부 정보를 나열하고 서비스 계정을 만들 수 있습니다.
- **Activity**: 각 user의 활동에 대한 세부 정보를 나열합니다.
- **Teams**: user 수 및 추적 시간을 포함하여 각 team에 대한 세부 정보를 나열하고 관리자가 team에 참여할 수 있도록 합니다.
- **Billing**: 조직의 요금을 요약하고, 결제 Reports를 실행 및 내보낼 수 있으며, 라이선스 만료 시기와 같은 세부 정보를 보여줍니다.
- **Settings**: 개인 정보 보호 및 인증과 관련된 사용자 정의 역할 및 설정을 구성할 수 있습니다.

## user 상태 보기
**Users** 탭에는 모든 user와 각 user에 대한 데이터가 나열되어 있습니다. **Last Active** 열은 user가 초대를 수락했는지 여부와 user의 현재 상태를 보여줍니다.

* **Invite pending**: 관리자가 초대를 보냈지만 user가 초대를 수락하지 않았습니다.
* **Active**: user가 초대를 수락하고 계정을 만들었습니다.
* **-**: user가 이전에 활성 상태였지만 지난 6개월 동안 활성 상태가 아닙니다.
* **Deactivated**: 관리자가 user의 엑세스를 취소했습니다.

활동별로 user 목록을 정렬하려면 **Last Active** 열 머리글을 클릭합니다.

## 조직에서 W&B를 사용하는 방법 보기 및 공유
**Users** 탭에서 조직에서 W&B를 사용하는 방법에 대한 세부 정보를 CSV 형식으로 볼 수 있습니다.

1. **Invite new user** 버튼 옆에 있는 작업 `...` 메뉴를 클릭합니다.
2. **Export as CSV**를 클릭합니다. 다운로드되는 CSV 파일에는 user 이름 및 이메일 어드레스, 마지막 활동 시간, 역할 등과 같은 조직의 각 user에 대한 세부 정보가 나열됩니다.

## user 활동 보기
**Users** 탭의 **Last Active** 열을 사용하여 개별 user의 **Activity summary**를 가져옵니다.

1. **Last Active**별로 user 목록을 정렬하려면 열 이름을 클릭합니다.
2. user의 마지막 활동에 대한 세부 정보를 보려면 user의 **Last Active** 필드 위에 마우스를 가져갑니다. user가 추가된 시기와 user가 총 며칠 동안 활성 상태였는지 보여주는 툴팁이 나타납니다.

다음과 같은 경우 user는 _active_ 상태입니다.
- W&B에 로그인합니다.
- W&B App에서 페이지를 봅니다.
- Runs를 로그합니다.
- SDK를 사용하여 experiment를 추적합니다.
- 어떤 방식으로든 W&B Server와 상호 작용합니다.

## 시간 경과에 따른 활성 user 보기
**Activity** 탭의 플롯을 사용하여 시간 경과에 따라 얼마나 많은 user가 활성 상태였는지에 대한 집계 보기를 얻을 수 있습니다.

1. **Activity** 탭을 클릭합니다.
2. **Total active users** 플롯은 특정 기간 동안 얼마나 많은 user가 활성 상태였는지 보여줍니다 (기본값은 3개월).
3. **Users active over time** 플롯은 특정 기간 동안 활성 user의 변동을 보여줍니다 (기본값은 6개월). 해당 날짜의 user 수를 보려면 포인트 위에 마우스를 가져갑니다.

플롯의 기간을 변경하려면 드롭다운을 사용합니다. 다음을 선택할 수 있습니다.
- Last 30 days
- Last 3 months
- Last 6 months
- Last 12 months
- All time
