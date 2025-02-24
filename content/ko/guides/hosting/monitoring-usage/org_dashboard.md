---
title: View organization dashboard
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-org_dashboard
    parent: monitoring-and-usage
---

{{% alert color="secondary" %}}
Organization dashboard는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 및 [자체 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed.md" >}})에서만 사용할 수 있습니다.
{{% /alert %}}

## W&B의 organization 사용량 보기
Organization dashboard를 사용하여 organization에 속한 Users, organization의 Users가 W&B를 사용하는 방식과 같은 속성에 대한 전체적인 뷰를 확인하십시오.

* **Name**: User 이름과 해당 W&B 사용자 이름.
* **Last active**: User가 마지막으로 W&B를 사용한 시간. 여기에는 제품 페이지 보기, Runs 기록, 기타 작업 수행 또는 로그인과 같이 인증이 필요한 모든 활동이 포함됩니다.
* **Role**: User의 역할.
* **Email**: User의 이메일.
* **Team**: User가 속한 Teams의 이름.

### User 상태 보기
**Last Active** 열은 User가 초대를 기다리는 중인지 아니면 활성 User인지 보여줍니다. User는 다음 세 가지 상태 중 하나입니다.

* **Invite pending**: 관리자가 초대를 보냈지만 User가 초대를 수락하지 않았습니다.
* **Active**: User가 초대를 수락하고 계정을 만들었습니다.
* **Deactivated**: 관리자가 User의 엑세스 권한을 취소했습니다.

{{< img src="/images/hosting/view_status_of_user.png" alt="" >}}

### organization의 W&B 사용 방식 보기 및 공유
organization에서 W&B를 사용하는 방식을 CSV 형식으로 봅니다.

1. **Add user** 버튼 옆에 있는 세 개의 점을 선택합니다.
2. 드롭다운에서 **Export as CSV**를 선택합니다.

    {{< img src="/images/hosting/export_org_usage.png" alt="" >}}

이렇게 하면 User 이름, 마지막 활동 타임스탬프, 역할, 이메일 등과 같은 User에 대한 세부 정보와 함께 organization의 모든 Users를 나열하는 CSV 파일이 내보내집니다.

### User 활동 보기
**Last Active** 열을 사용하여 개별 User의 **Activity summary**를 가져옵니다.

1. User의 **Last Active** 항목 위에 마우스 커서를 올려 놓습니다.
2. 툴팁이 나타나 User 활동에 대한 정보 요약을 제공합니다.

{{< img src="/images/hosting/activity_tooltip.png" alt="" >}}

User는 다음과 같은 경우 _active_ 상태입니다.
- W&B에 로그인합니다.
- W&B 앱에서 페이지를 봅니다.
- Runs를 기록합니다.
- SDK를 사용하여 Experiments를 추적합니다.
- 어떤 방식으로든 W&B Server와 상호 작용합니다.

### 시간 경과에 따른 활성 Users 보기
Organization dashboard의 **Users active over time** 플롯을 사용하여 시간 경과에 따른 활성 Users 수를 집계하여 개요를 확인합니다(아래 이미지에서 가장 오른쪽 플롯).

{{< img src="/images/hosting/dashboard_summary.png" alt="" >}}

드롭다운 메뉴를 사용하여 결과 필터링 기준을 일, 월 또는 전체 시간으로 설정할 수 있습니다.
