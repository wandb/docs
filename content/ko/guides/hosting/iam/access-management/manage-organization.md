---
title: Manage your organization
menu:
  default:
    identifier: ko-guides-hosting-iam-access-management-manage-organization
    parent: access-management
weight: 1
---

Organizations의 관리자는 Organization 내에서 [개별 사용자를 관리]({{< relref path="#add-and-manage-users" lang="ko" >}})하고 [Teams를 관리]({{< relref path="#add-and-manage-teams" lang="ko" >}})할 수 있습니다.

Team 관리자는 [Teams를 관리]({{< relref path="#add-and-manage-teams" lang="ko" >}})할 수 있습니다.

{{% alert %}}
다음 워크플로우는 인스턴스 관리자 역할을 가진 사용자에게 적용됩니다. 인스턴스 관리자 권한이 있어야 한다고 생각되면 Organization의 관리자에게 문의하십시오.
{{% /alert %}}

Organization에서 사용자 관리를 간소화하려면 [사용자 및 Team 관리 자동화]({{< relref path="../automate_iam.md" lang="ko" >}})를 참조하십시오.

## Organization 이름 변경

{{% alert %}}
다음 워크플로우는 W&B Multi-tenant SaaS Cloud에만 적용됩니다.
{{% /alert %}}

1. https://wandb.ai/home 으로 이동합니다.
2. 페이지 오른쪽 상단에서 **User menu** 드롭다운을 선택합니다. 드롭다운의 **Account** 섹션에서 **Settings**를 선택합니다.
3. **Settings** 탭에서 **General**을 선택합니다.
4. **Change name** 버튼을 선택합니다.
5. 나타나는 모달에서 Organization의 새 이름을 입력하고 **Save name** 버튼을 선택합니다.

## 사용자 추가 및 관리

관리자는 Organization의 대시보드를 사용하여 다음을 수행할 수 있습니다.
- 사용자 초대 또는 제거
- 사용자의 Organization 역할 할당 또는 업데이트, 사용자 지정 역할 생성
- 결제 관리자 할당

Organization 관리자가 사용자를 Organization에 추가할 수 있는 방법은 여러 가지가 있습니다.

1. Member-by-invite
2. SSO를 사용한 자동 프로비저닝
3. 도메인 캡처

### Seats 및 가격

다음 표는 Models 및 Weave의 Seats 작동 방식을 요약한 것입니다.

| 제품 | Seats | 비용 기준 |
| ----- | ----- | ----- |
| Models | 설정당 지불 | 보유한 Models 유료 Seats 수와 발생한 사용량에 따라 전체 구독 비용이 결정됩니다. 각 사용자는 Full, Viewer, No-Access의 세 가지 사용 가능한 Seat 유형 중 하나를 할당받을 수 있습니다. |
| Weave | 무료 | 사용량 기준 |

### 사용자 초대

관리자는 사용자를 Organization 내의 특정 Teams뿐만 아니라 Organization에 초대할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="Multi-tenant SaaS Cloud" value="saas" %}}
1. https://wandb.ai/home 으로 이동합니다.
1. 페이지 오른쪽 상단에서 **User menu** 드롭다운을 선택합니다. 드롭다운의 **Account** 섹션에서 **Users**를 선택합니다.
3. **Invite new user**를 선택합니다.
4. 나타나는 모달에서 **Email or username** 필드에 사용자의 이메일 또는 사용자 이름을 입력합니다.
5. (권장) **Choose teams** 드롭다운 메뉴에서 사용자를 Team에 추가합니다.
6. **Select role** 드롭다운에서 사용자에게 할당할 역할을 선택합니다. 사용자의 역할은 나중에 변경할 수 있습니다. 가능한 역할에 대한 자세한 내용은 [역할 할당]({{< relref path="#assign-or-update-a-team-members-role" lang="ko" >}})에 나열된 표를 참조하십시오.
7. **Send invite** 버튼을 선택합니다.

**Send invite** 버튼을 선택하면 W&B에서 타사 이메일 서버를 사용하여 사용자 이메일로 초대 링크를 보냅니다. 사용자는 초대를 수락하면 Organization에 액세스할 수 있습니다.
{{% /tab %}}

{{% tab header="Dedicated or Self-managed" value="dedicated"%}}
1. `https://<org-name>.io/console/settings/`로 이동합니다. `<org-name>`을 Organization 이름으로 바꿉니다.
2. **Add user** 버튼을 선택합니다.
3. 나타나는 모달에서 **Email** 필드에 새 사용자의 이메일을 입력합니다.
4. **Role** 드롭다운에서 사용자에게 할당할 역할을 선택합니다. 사용자의 역할은 나중에 변경할 수 있습니다. 가능한 역할에 대한 자세한 내용은 [역할 할당]({{< relref path="#assign-or-update-a-team-members-role" lang="ko" >}})에 나열된 표를 참조하십시오.
5. W&B에서 타사 이메일 서버를 사용하여 사용자 이메일로 초대 링크를 보내려면 **Send invite email to user** 상자를 선택합니다.
6. **Add new user** 버튼을 선택합니다.
{{% /tab %}}
{{< /tabpane >}}

### 사용자 자동 프로비저닝

SSO를 구성하고 SSO 제공업체에서 허용하는 경우 일치하는 이메일 도메인을 가진 W&B 사용자는 Single Sign-On (SSO)으로 W&B Organization에 로그인할 수 있습니다. SSO는 모든 Enterprise 라이선스에서 사용할 수 있습니다.

{{% alert title="인증을 위해 SSO 활성화" %}}
W&B는 사용자가 Single Sign-On (SSO)을 사용하여 인증할 것을 강력히 권장합니다. Organization에 대해 SSO를 활성화하려면 W&B Team에 문의하십시오.

Dedicated Cloud 또는 Self-managed 인스턴스로 SSO를 설정하는 방법에 대한 자세한 내용은 [OIDC를 사용한 SSO]({{< relref path="../authentication/sso.md" lang="ko" >}}) 또는 [LDAP를 사용한 SSO]({{< relref path="../authentication/ldap.md" lang="ko" >}})를 참조하십시오.{{% /alert %}}

W&B는 자동 프로비저닝 사용자에게 기본적으로 "Member" 역할을 할당합니다. 자동 프로비저닝 사용자의 역할은 언제든지 변경할 수 있습니다.

SSO를 통한 자동 프로비저닝 사용자는 Dedicated Cloud 인스턴스 및 Self-managed 배포에서 기본적으로 켜져 있습니다. 자동 프로비저닝을 끌 수 있습니다. 자동 프로비저닝을 끄면 특정 사용자를 W&B Organization에 선택적으로 추가할 수 있습니다.

다음 탭에서는 배포 유형에 따라 SSO를 끄는 방법을 설명합니다.

{{< tabpane text=true >}}
{{% tab header="Dedicated cloud" value="dedicated" %}}
Dedicated Cloud 인스턴스를 사용 중이고 SSO를 통한 자동 프로비저닝을 끄려면 W&B Team에 문의하십시오.
{{% /tab %}}

{{% tab header="Self-managed" value="self_managed" %}}
W&B Console을 사용하여 SSO를 통한 자동 프로비저닝을 끕니다.

1. `https://<org-name>.io/console/settings/`로 이동합니다. `<org-name>`을 Organization 이름으로 바꿉니다.
2. **Security**를 선택합니다.
3. **Disable SSO Provisioning**을 선택하여 SSO를 통한 자동 프로비저닝을 끕니다.

{{% /tab %}}
{{< /tabpane >}}

{{% alert title="" %}}
SSO를 통한 자동 프로비저닝은 Organization 관리자가 개별 사용자 초대를 생성할 필요가 없기 때문에 대규모로 사용자를 Organization에 추가하는 데 유용합니다.
{{% /alert %}}

### 사용자 지정 역할 만들기

{{% alert %}}
Dedicated Cloud 또는 Self-managed 배포에서 사용자 지정 역할을 만들거나 할당하려면 Enterprise 라이선스가 필요합니다.
{{% /alert %}}

Organization 관리자는 View-Only 또는 Member 역할을 기반으로 새 역할을 구성하고 세분화된 엑세스 제어를 위해 추가 권한을 추가할 수 있습니다. Team 관리자는 Team member에게 사용자 지정 역할을 할당할 수 있습니다. 사용자 지정 역할은 Organization 수준에서 생성되지만 Team 수준에서 할당됩니다.

사용자 지정 역할을 만들려면:

{{< tabpane text=true >}}
{{% tab header="Multi-tenant SaaS Cloud" value="saas" %}}
1. https://wandb.ai/home 으로 이동합니다.
1. 페이지 오른쪽 상단에서 **User menu** 드롭다운을 선택합니다. 드롭다운의 **Account** 섹션에서 **Settings**를 선택합니다.
1. **Roles**를 클릭합니다.
1. **Custom roles** 섹션에서 **Create a role**을 클릭합니다.
1. 역할 이름을 입력합니다. 선택적으로 설명을 입력합니다.
1. 사용자 지정 역할의 기반으로 사용할 역할을 선택합니다 (**Viewer** 또는 **Member**).
1. 권한을 추가하려면 **Search permissions** 필드를 클릭한 다음 추가할 권한을 하나 이상 선택합니다.
1. 역할에 있는 권한을 요약하는 **Custom role permissions** 섹션을 검토합니다.
1. **Create Role**을 클릭합니다.
{{% /tab %}}

{{% tab header="Dedicated or Self-managed" value="dedicated"%}}
W&B Console을 사용하여 SSO를 통한 자동 프로비저닝을 끕니다.

1. `https://<org-name>.io/console/settings/`로 이동합니다. `<org-name>`을 Organization 이름으로 바꿉니다.
1. **Custom roles** 섹션에서 **Create a role**을 클릭합니다.
1. 역할 이름을 입력합니다. 선택적으로 설명을 입력합니다.
1. 사용자 지정 역할의 기반으로 사용할 역할을 선택합니다 (**Viewer** 또는 **Member**).
1. 권한을 추가하려면 **Search permissions** 필드를 클릭한 다음 추가할 권한을 하나 이상 선택합니다.
1. 역할에 있는 권한을 요약하는 **Custom role permissions** 섹션을 검토합니다.
1. **Create Role**을 클릭합니다.

{{% /tab %}}
{{< /tabpane >}}

이제 Team 관리자는 [Team 설정]({{< relref path="#invite-users-to-a-team" lang="ko" >}})에서 Team의 구성원에게 사용자 지정 역할을 할당할 수 있습니다.

### 도메인 캡처

도메인 캡처는 직원이 회사 Organization에 가입하여 새 사용자가 회사 관할 구역 외부에서 에셋을 생성하지 않도록 하는 데 도움이 됩니다.

{{% alert title="도메인은 고유해야 합니다." %}}
도메인은 고유한 식별자입니다. 즉, 다른 Organization에서 이미 사용 중인 도메인은 사용할 수 없습니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="Multi-tenant SaaS Cloud" value="saas" %}}
도메인 캡처를 사용하면 `@example.com`과 같은 회사 이메일 주소를 가진 사람들을 W&B SaaS Cloud Organization에 자동으로 추가할 수 있습니다. 이는 모든 직원이 올바른 Organization에 가입하고 새 사용자가 회사 관할 구역 외부에서 에셋을 생성하지 않도록 하는 데 도움이 됩니다.

다음 표는 도메인 캡처 활성화 여부에 따른 신규 및 기존 사용자의 행동을 요약한 것입니다.

| | 도메인 캡처 사용 | 도메인 캡처 사용 안 함 |
| ----- | ----- | ----- |
| 신규 사용자 | 확인된 도메인에서 W&B에 가입하는 사용자는 Organization의 기본 Team에 자동으로 구성원으로 추가됩니다. Team 가입을 활성화하면 가입 시 추가 Team을 선택할 수 있습니다. 초대장을 통해 다른 Organization 및 Team에 가입할 수도 있습니다. | 사용자는 사용 가능한 중앙 집중식 Organization이 있는지 모르고 W&B 계정을 만들 수 있습니다. |
| 초대된 사용자 | 초대된 사용자는 초대를 수락하면 자동으로 Organization에 가입합니다. 초대된 사용자는 Organization의 기본 Team에 자동으로 구성원으로 추가되지 않습니다. 초대장을 통해 다른 Organization 및 Team에 가입할 수도 있습니다. | 초대된 사용자는 초대를 수락하면 자동으로 Organization에 가입합니다. 초대장을 통해 다른 Organization 및 Team에 가입할 수도 있습니다. |
| 기존 사용자 | 도메인의 확인된 이메일 주소를 가진 기존 사용자는 W&B 앱 내에서 Organization의 Teams에 가입할 수 있습니다. 기존 사용자가 Organization에 가입하기 전에 생성한 모든 데이터는 유지됩니다. W&B는 기존 사용자의 데이터를 마이그레이션하지 않습니다. | 기존 W&B 사용자는 여러 Organization 및 Teams에 분산될 수 있습니다. |

초대받지 않은 신규 사용자가 Organization에 가입할 때 기본 Team에 자동으로 할당하려면:

1. https://wandb.ai/home 으로 이동합니다.
2. 페이지 오른쪽 상단에서 **User menu** 드롭다운을 선택합니다. 드롭다운에서 **Settings**를 선택합니다.
3. **Settings** 탭에서 **General**을 선택합니다.
4. **Domain capture** 내에서 **Claim domain** 버튼을 선택합니다.
5. **Default team** 드롭다운에서 새 사용자를 자동으로 가입시킬 Team을 선택합니다. 사용 가능한 Team이 없으면 Team 설정을 업데이트해야 합니다. [Teams 추가 및 관리]({{< relref path="#add-and-manage-teams" lang="ko" >}})의 지침을 참조하십시오.
6. **Claim email domain** 버튼을 클릭합니다.

초대받지 않은 신규 사용자를 Team에 자동으로 할당하려면 먼저 Team 설정 내에서 도메인 일치를 활성화해야 합니다.

1. `https://wandb.ai/<team-name>`에서 Team의 대시보드로 이동합니다. 여기서 `<team-name>`은 도메인 일치를 활성화할 Team의 이름입니다.
2. Team 대시보드의 왼쪽 탐색 모음에서 **Team settings**를 선택합니다.
3. **Privacy** 섹션 내에서 "가입 시 일치하는 이메일 도메인을 가진 새 사용자가 이 Team에 가입하도록 추천" 옵션을 토글합니다.
{{% /tab %}}
{{% tab header="Dedicated or Self-managed" value="dedicated" %}}
도메인 캡처를 구성하려면 Dedicated 또는 Self-managed 배포 유형을 사용하는 경우 W&B Account Team에 문의하십시오. 구성되면 W&B SaaS 인스턴스는 회사 이메일 주소로 W&B 계정을 만드는 사용자에게 관리자에게 연락하여 Dedicated 또는 Self-managed 인스턴스에 대한 엑세스를 요청하라는 메시지를 자동으로 표시합니다.

| | 도메인 캡처 사용 | 도메인 캡처 사용 안 함 |
| ----- | ----- | -----|
| 신규 사용자 | 확인된 도메인에서 SaaS Cloud에서 W&B에 가입하는 사용자는 사용자 지정하는 이메일 주소로 관리자에게 연락하라는 메시지가 자동으로 표시됩니다. SaaS Cloud에서 Organizations을 만들어 제품을 트라이얼할 수도 있습니다. | 사용자는 회사에 중앙 집중식 Dedicated 인스턴스가 있다는 것을 모른 채 W&B SaaS Cloud 계정을 만들 수 있습니다. |
| 기존 사용자 | 기존 W&B 사용자는 여러 Organization 및 Teams에 분산될 수 있습니다. | 기존 W&B 사용자는 여러 Organization 및 Teams에 분산될 수 있습니다. |
{{% /tab %}}
{{< /tabpane >}}

### 사용자의 역할 할당 또는 업데이트

Organization의 모든 구성원은 W&B Models 및 Weave 모두에 대한 Organization 역할과 Seat를 갖습니다. 그들이 가지고 있는 Seat 유형은 그들의 결제 상태와 각 제품 라인에서 그들이 할 수 있는 작업을 결정합니다.

사용자를 Organization에 초대할 때 처음으로 Organization 역할을 할당합니다. 나중에 사용자의 역할을 변경할 수 있습니다.

Organization 내의 사용자는 다음 역할 중 하나를 가질 수 있습니다.

| 역할 | 설명 |
| ----- | ----- |
| admin | 다른 사용자를 Organization에 추가하거나 제거하고, 사용자 역할을 변경하고, 사용자 지정 역할을 관리하고, Teams를 추가할 수 있는 인스턴스 관리자입니다. W&B는 관리자가 부재중인 경우를 대비하여 둘 이상의 관리자가 있는지 확인하는 것이 좋습니다. |
| Member | 인스턴스 관리자가 초대한 Organization의 일반 사용자입니다. Organization member는 다른 사용자를 초대하거나 Organization의 기존 사용자를 관리할 수 없습니다. |
| Viewer (Enterprise 전용 기능) | 인스턴스 관리자가 초대한 Organization의 보기 전용 사용자입니다. Viewer는 Organization과 그들이 구성원인 기본 Teams에 대한 읽기 엑세스 권한만 있습니다. |
| 사용자 지정 역할 (Enterprise 전용 기능) | 사용자 지정 역할을 통해 Organization 관리자는 이전 View-Only 또는 Member 역할에서 상속하고 세분화된 엑세스 제어를 위해 추가 권한을 추가하여 새 역할을 구성할 수 있습니다. 그런 다음 Team 관리자는 해당 사용자 지정 역할을 각 Teams의 사용자에게 할당할 수 있습니다. |

사용자의 역할을 변경하려면:

1. https://wandb.ai/home 으로 이동합니다.
2. 페이지 오른쪽 상단에서 **User menu** 드롭다운을 선택합니다. 드롭다운에서 **Users**를 선택합니다.
4. 검색 창에 사용자의 이름 또는 이메일을 입력합니다.
4. 사용자 이름 옆에 있는 **TEAM ROLE** 드롭다운에서 역할을 선택합니다.

### 사용자 엑세스 할당 또는 업데이트

Organization 내의 사용자는 다음 Models Seat 또는 Weave 엑세스 유형 중 하나를 갖습니다. full, viewer 또는 no access.

| Seat 유형 | 설명 |
| ----- | ----- |
| Full | 이 역할 유형을 가진 사용자는 Models 또는 Weave에 대한 데이터를 쓰고, 읽고, 내보낼 수 있는 모든 권한을 갖습니다. |
| Viewer | Organization의 보기 전용 사용자입니다. Viewer는 Organization과 그들이 속한 기본 Teams에 대한 읽기 엑세스 권한만 있고 Models 또는 Weave에 대한 보기 전용 엑세스 권한만 있습니다. |
| No access | 이 역할을 가진 사용자는 Models 또는 Weave 제품에 대한 엑세스 권한이 없습니다. |

Model Seat 유형 및 Weave 엑세스 유형은 Organization 수준에서 정의되며 Team에서 상속됩니다. 사용자의 Seat 유형을 변경하려면 Organization 설정으로 이동하여 다음 단계를 따르십시오.

1. SaaS 사용자의 경우 `https://wandb.ai/account-settings/<organization>/settings`에서 Organization 설정으로 이동합니다. 꺾쇠 괄호 (`<>`)로 묶인 값을 Organization 이름으로 바꿔야 합니다. 다른 Dedicated 및 Self-managed 배포의 경우 `https://<your-instance>.wandb.io/org/dashboard`로 이동합니다.
2. **Users** 탭을 선택합니다.
3. **Role** 드롭다운에서 사용자에게 할당할 Seat 유형을 선택합니다.

{{% alert %}}
Organization 역할과 구독 유형에 따라 Organization 내에서 사용할 수 있는 Seat 유형이 결정됩니다.
{{% /alert %}}

### 사용자 제거

1. https://wandb.ai/home 으로 이동합니다.
2. 페이지 오른쪽 상단에서 **User menu** 드롭다운을 선택합니다. 드롭다운에서 **Users**를 선택합니다.
4. 검색 창에 사용자의 이름 또는 이메일을 입력합니다.
5. 나타날 때 줄임표 또는 세 개의 점 아이콘 (**...**)을 선택합니다.
6. 드롭다운에서 **Remove member**를 선택합니다.

### 결제 관리자 할당

1. https://wandb.ai/home 으로 이동합니다.
2. 페이지 오른쪽 상단에서 **User menu** 드롭다운을 선택합니다. 드롭다운에서 **Users**를 선택합니다.
4. 검색 창에 사용자의 이름 또는 이메일을 입력합니다.
5. **Billing admin** 열에서 결제 관리자로 할당할 사용자를 선택합니다.

## Teams 추가 및 관리

Organization 대시보드를 사용하여 Organization 내에서 Teams를 만들고 관리합니다. Organization 관리자 또는 Team 관리자는 다음을 수행할 수 있습니다.
- 사용자를 Team에 초대하거나 Team에서 사용자를 제거합니다.
- Team member의 역할을 관리합니다.
- 사용자가 Organization에 가입할 때 Team에 사용자를 자동으로 추가합니다.
- `https://wandb.ai/<team-name>`에서 Team의 대시보드를 사용하여 Team 스토리지를 관리합니다.

### Team 만들기

Organization 대시보드를 사용하여 Team을 만듭니다.

1. https://wandb.ai/home 으로 이동합니다.
2. 왼쪽 네비게이션 패널의 **Teams** 아래에서 **Create a team to collaborate**를 선택합니다.
{{< img src="/images/hosting/create_new_team.png" alt="" >}}
3. 나타나는 모달에서 **Team name** 필드에 Team 이름을 입력합니다.
4. 스토리지 유형을 선택합니다.
5. **Create team** 버튼을 선택합니다.

**Create team** 버튼을 선택하면 W&B에서 `https://wandb.ai/<team-name>`의 새 Team 페이지로 리디렉션합니다. 여기서 `<team-name>`은 Team을 만들 때 제공하는 이름으로 구성됩니다.

Team이 있으면 해당 Team에 사용자를 추가할 수 있습니다.

### Team에 사용자 초대

Organization에서 Team에 사용자를 초대합니다. Team의 대시보드를 사용하여 이메일 주소 또는 W&B 사용자 이름(이미 W&B 계정이 있는 경우)을 사용하여 사용자를 초대합니다.

1. `https://wandb.ai/<team-name>`로 이동합니다.
2. 대시보드의 왼쪽 글로벌 네비게이션에서 **Team settings**를 선택합니다.
{{< img src="/images/hosting/team_settings.png" alt="" >}}
3. **Users** 탭을 선택합니다.
4. **Invite a new user**를 선택합니다.
5. 나타나는 모달에서 **Email or username** 필드에 사용자의 이메일을 입력하고 **Select a team** 역할 드롭다운에서 해당 사용자에게 할당할 역할을 선택합니다. 사용자가 Team에서 가질 수 있는 역할에 대한 자세한 내용은 [Team 역할]({{< relref path="#assign-or-update-a-team-members-role" lang="ko" >}})을 참조하십시오.
6. **Send invite** 버튼을 클릭합니다.

기본적으로 Team 또는 인스턴스 관리자만 Team에 구성원을 초대할 수 있습니다. 이 동작을 변경하려면 [Team 설정]({{< relref path="/guides/models/app/settings-page/team-settings.md#privacy" lang="ko" >}})을 참조하십시오.

이메일 초대를 통해 수동으로 사용자를 초대하는 것 외에도 새 사용자의 [이메일이 Organization의 도메인과 일치하는 경우]({{< relref path="#domain-capture" lang="ko" >}}) 새 사용자를 Team에 자동으로 추가할 수 있습니다.

### 가입 시 Team Organization에 구성원 일치

새 사용자가 가입할 때 Organization 내에서 Teams를 검색할 수 있도록 허용합니다. 새 사용자는 Organization의 확인된 이메일 도메인과 일치하는 확인된 이메일 도메인이 있어야 합니다. 확인된 새 사용자는 W&B 계정에 가입할 때 Organization에 속한 확인된 Teams 목록을 볼 수 있습니다.

Organization 관리자는 도메인 클레임을 활성화해야 합니다. 도메인 캡처를 활성화하려면 [도메인 캡처]({{< relref path="#domain-capture" lang="ko" >}})에 설명된 단계를 참조하십시오.

### Team member의 역할 할당 또는 업데이트

1. Team member 이름 옆에 있는 계정 유형 아이콘을 선택합니다.
2. 드롭다운에서 해당 Team member가 가질 계정 유형을 선택합니다.

다음 표는 Team 구성원에게 할당할 수 있는 역할을 나열합니다.

| 역할 | 정의 |
|-----------|---------------------------|
| admin | Team에서 다른 사용자를 추가 및 제거하고, 사용자 역할을 변경하고, Team 설정을 구성할 수 있는 사용자입니다. |
| Member | Team 관리자가 이메일 또는 Organization 수준 사용자 이름으로 초대한 Team의 일반 사용자입니다. member 사용자는 다른 사용자를 Team에 초대할 수 없습니다. |
| View-Only (Enterprise 전용 기능) | Team 관리자가 이메일 또는 Organization 수준 사용자 이름으로 초대한 Team의 보기 전용 사용자입니다. 보기 전용 사용자는 Team과 그 내용에 대한 읽기 엑세스 권한만 있습니다. |
| Service (Enterprise 전용 기능) | 서비스 작업자 또는 서비스 계정은 run 자동화 툴로 W&B를 활용하는 데 유용한 API 키입니다. Team의 서비스 계정에서 API 키를 사용하는 경우 환경 변수 `WANDB_USERNAME`을 설정하여 run을 적절한 사용자에게 올바르게 할당해야 합니다. |
| 사용자 지정 역할 (Enterprise 전용 기능) | 사용자 지정 역할을 통해 Organization 관리자는 이전 View-Only 또는 Member 역할에서 상속하고 세분화된 엑세스 제어를 위해 추가 권한을 추가하여 새 역할을 구성할 수 있습니다. 그런 다음 Team 관리자는 해당 사용자 지정 역할을 각 Teams의 사용자에게 할당할 수 있습니다. 자세한 내용은 [이 문서](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3)를 참조하십시오. |

{{% alert %}}
Dedicated Cloud 또는 Self-managed 배포의 Enterprise 라이선스만 Team 구성원에게 사용자 지정 역할을 할당할 수 있습니다.
{{% /alert %}}

### Team에서 사용자 제거

Team의 대시보드를 사용하여 Team에서 사용자를 제거합니다. W&B는 run을 생성한 구성원이 더 이상 해당 Team에 없더라도 Team에서 생성된 run을 보존합니다.

1. `https://wandb.ai/<team-name>`로 이동합니다.
2. 왼쪽 네비게이션 바에서 **Team settings**를 선택합니다.
3. **Users** 탭을 선택합니다.
4. 삭제할 사용자 이름 옆에 마우스를 가져갑니다. 나타날 때 줄임표 또는 세 개의 점 아이콘 (**...**)을 선택합니다.
5. 드롭다운에서 **Remove user**를 선택합니다.
