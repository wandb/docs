---
title: 조직 관리하기
menu:
  default:
    identifier: ko-guides-hosting-iam-access-management-manage-organization
    parent: access-management
weight: 1
---

조직의 관리자인 경우, 조직 내에서 [개별 사용자 관리]({{< relref path="#add-and-manage-users" lang="ko" >}}) 와 [Teams 관리]({{< relref path="#add-and-manage-teams" lang="ko" >}}) 를 할 수 있습니다.

팀 관리자인 경우, [Teams 관리]({{< relref path="#add-and-manage-teams" lang="ko" >}}) 가 가능합니다.

{{% alert %}}
다음 워크플로우는 인스턴스 관리자 역할을 가진 사용자에게 적용됩니다. 본인이 인스턴스 관리자 권한이 필요하다고 생각되면 조직 내 관리자에게 문의하세요.
{{% /alert %}}

조직에서 사용자 관리를 더 간단하게 하고 싶다면 [사용자 및 팀 관리 자동화]({{< relref path="../automate_iam.md" lang="ko" >}}) 를 참고하세요.




## 조직 이름 변경하기
{{% alert %}}
다음 워크플로우는 W&B 멀티 테넌트 SaaS 클라우드에서만 적용됩니다.
{{% /alert %}}

1. https://wandb.ai/home 으로 이동합니다.
2. 우측 상단의 **User menu** 드롭다운을 클릭합니다. 드롭다운의 **Account** 섹션에서 **Settings** 를 선택하세요.
3. **Settings** 탭에서 **General** 을 선택합니다.
4. **Change name** 버튼을 선택합니다.
5. 나타나는 모달에서 새로운 조직명을 입력하고 **Save name** 버튼을 선택합니다.

## 사용자 추가 및 관리

관리자는 조직의 대시보드를 사용해 다음을 할 수 있습니다:
- 사용자 초대 또는 제거
- 사용자의 조직 내 역할 부여 또는 수정, 커스텀 역할 생성
- 결제 관리자 지정

조직 관리자는 여러 방법으로 조직에 사용자를 추가할 수 있습니다:

1. 초대 기반 멤버 추가
2. SSO를 통한 자동 프로비저닝
3. 도메인 캡처

### 좌석 및 가격 정책

다음 표는 Models 와 Weave의 좌석 정책을 요약합니다:

| Product |좌석 수 | 비용 기준 |
| ----- | ----- | ----- |
| Models | 좌석 별 결제 | 보유한 Models 유료 좌석 수와 사용량에 따라 구독 비용이 결정됩니다. 각 사용자는 Full, Viewer, No-Access 중 하나의 좌석 유형을 할당받을 수 있습니다. |
| Weave | 무료  | 사용량 기반 |

### 사용자 초대

관리자는 조직 또는 해당 조직 내 특정 Teams에 사용자를 초대할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="멀티 테넌트 SaaS 클라우드" value="saas" %}}
1. https://wandb.ai/home 으로 이동합니다.
1. 우측 상단의 **User menu** 드롭다운을 클릭합니다. 드롭다운의 **Account** 섹션에서 **Users** 를 선택합니다.
3. **Invite new user** 를 선택합니다.
4. 나타나는 모달에서 **Email or username** 입력란에 사용자의 이메일 또는 사용자명을 입력합니다.
5. (권장) **Choose teams** 드롭다운에서 사용자를 추가할 팀을 선택합니다.
6. **Select role** 드롭다운에서 사용자에게 할당할 역할을 선택합니다. 사용자의 역할은 추후 변경할 수 있습니다. 가능한 역할에 대한 자세한 내용은 [Assign a role]({{< relref path="#assign-or-update-a-team-members-role" lang="ko" >}}) 표를 참고하세요.
7. **Send invite** 버튼을 클릭합니다.

**Send invite** 버튼을 클릭하면, W&B가 서드파티 이메일 서버를 통해 해당 사용자에게 초대 링크를 발송합니다. 사용자가 초대장을 수락하면 귀하의 조직에 바로 엑세스할 수 있습니다.
{{% /tab %}}

{{% tab header="전용 또는 셀프 매니지드" value="dedicated"%}}
1. `https://<org-name>.io/console/settings/` 로 이동합니다. `<org-name>` 부분에는 귀하의 조직명을 입력하세요.
2. **Add user** 버튼을 클릭합니다.
3. 나타나는 모달에서 새로운 사용자의 **Email** 입력란에 이메일을 입력합니다.
4. **Role** 드롭다운에서 사용자에게 할당할 역할을 선택합니다. 사용자의 역할은 추후 변경이 가능합니다. 가능한 역할에 대한 자세한 내용은 [Assign a role]({{< relref path="#assign-or-update-a-team-members-role" lang="ko" >}}) 표를 참고하세요.
5. W&B에서 서드파티 이메일 서버를 통해 초대메일을 발송하려면 **Send invite email to user** 체크박스를 활성화합니다.
6. **Add new user** 버튼을 클릭합니다.
{{% /tab %}}
{{< /tabpane >}}

### 사용자 자동 프로비저닝

W&B 사용자는 이메일 도메인이 일치하는 경우 Single Sign-On(SSO) 설정을 통해 W&B Organization 에 로그인할 수 있습니다. SSO는 모든 엔터프라이즈 라이선스에서 제공됩니다.

{{% alert title="인증을 위한 SSO 활성화" %}}
W&B에서는 사용자 인증 시 Single Sign-On(SSO) 사용을 강력히 권장합니다. 조직에 SSO 활성화를 원한다면 W&B 팀에 문의하세요.

Dedicated 클라우드 또는 셀프 매니지드 인스턴스에서 SSO 설정 방법은 [OIDC 기반 SSO]({{< relref path="../authentication/sso.md" lang="ko" >}}) 또는 [LDAP 기반 SSO]({{< relref path="../authentication/ldap.md" lang="ko" >}}) 를 참고하세요.{{% /alert %}}

W&B는 자동 프로비저닝된 사용자에게 기본적으로 "Member" 역할을 할당합니다. 자동 프로비저닝된 사용자의 역할은 언제든지 변경할 수 있습니다.

Dedicated 클라우드 인스턴스 및 셀프 매니지드 배포에서는 기본적으로 SSO를 통한 자동 프로비저닝이 활성화되어 있습니다. 자동 프로비저닝을 비활성화하면 특정 사용자를 선별적으로 W&B 조직에 추가할 수 있습니다.

아래 탭별로 배포 방식에 따른 SSO 비활성화 방법을 설명합니다.

{{< tabpane text=true >}}
{{% tab header="전용 클라우드" value="dedicated" %}}
전용 클라우드 인스턴스를 사용 중이며 SSO 자동 프로비저닝을 끄고 싶다면, W&B 팀에 문의하세요.
{{% /tab %}}

{{% tab header="셀프 매니지드" value="self_managed" %}}
W&B 콘솔을 사용해 SSO 자동 프로비저닝을 끌 수 있습니다.

1. `https://<org-name>.io/console/settings/`로 이동합니다. `<org-name>`에는 조직명을 입력하세요.
2. **Security** 를 선택합니다.
3. **Disable SSO Provisioning** 을 선택해 SSO 자동 프로비저닝을 끕니다.




{{% /tab %}}
{{< /tabpane >}}

{{% alert title="" %}}
SSO를 통한 자동 프로비저닝은 대규모로 조직에 사용자를 추가할 때 편리합니다. 조직 관리자가 개별적으로 초대장을 발송할 필요가 없습니다.
{{% /alert %}}

### 커스텀 역할 생성
{{% alert %}}
Dedicated 클라우드 또는 셀프 매니지드 배포에서 커스텀 역할을 생성/할당하려면 엔터프라이즈 라이선스가 필요합니다.
{{% /alert %}}

조직 관리자는 View-Only 또는 Member 역할을 기반으로 하여 추가 권한을 부여하는 새로운 역할을 만들 수 있습니다. 팀 관리자는 커스텀 역할을 팀 멤버에게 할당할 수 있습니다. 커스텀 역할은 조직 레벨에서 생성되지만, 팀 레벨에서 할당됩니다.

커스텀 역할을 생성하려면:

{{< tabpane text=true >}}
{{% tab header="멀티 테넌트 SaaS 클라우드" value="saas" %}}
1. https://wandb.ai/home 에 접속하세요.
1. 우측 상단의 **User menu** 드롭다운을 클릭합니다. **Account** 섹션에서 **Settings** 를 선택하세요.
1. **Roles** 를 클릭합니다.
1. **Custom roles** 섹션에서 **Create a role** 을 클릭합니다.
1. 역할 이름을 입력하세요. 원하면 설명도 입력할 수 있습니다.
1. **Viewer** 또는 **Member** 중에서 커스텀 역할의 기반이 되는 역할을 선택하세요.
1. 권한을 추가하려면 **Search permissions** 입력란을 클릭하고, 추가할 권한을 하나 이상 선택하세요.
1. **Custom role permissions** 섹션에서 해당 역할의 권한 요약을 확인하세요.
1. **Create Role** 을 클릭하세요.
{{% /tab %}}

{{% tab header="전용 또는 셀프 매니지드" value="dedicated"%}}
W&B 콘솔을 사용해 커스텀 역할을 생성하세요.

1. `https://<org-name>.io/console/settings/`로 이동합니다. `<org-name>`에는 조직명을 입력하세요.
1. **Custom roles** 섹션에서 **Create a role** 을 클릭합니다.
1. 역할 이름을 입력하세요. 원하면 설명도 입력할 수 있습니다.
1. **Viewer** 또는 **Member** 중에서 커스텀 역할의 기반이 되는 역할을 선택하세요.
1. 권한을 추가하려면 **Search permissions** 입력란을 클릭하고, 추가할 권한을 하나 이상 선택하세요.
1. **Custom role permissions** 섹션에서 해당 역할의 권한 요약을 확인하세요.
1. **Create Role** 을 클릭하세요.

{{% /tab %}}
{{< /tabpane >}}

이제 팀 관리자는 [Team settings]({{< relref path="#invite-users-to-a-team" lang="ko" >}}) 에서 커스텀 역할을 팀 멤버에게 할당할 수 있습니다.

### 도메인 캡처
도메인 캡처를 사용하면 임직원이 회사 조직에 쉽게 참여하고, 신규 사용자가 회사 관할 밖에서 자산을 생성하지 않도록 할 수 있습니다.

{{% alert title="도메인은 고유해야 합니다" %}}
도메인은 고유 식별자입니다. 이미 다른 조직에서 사용 중인 도메인은 사용할 수 없습니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="멀티 테넌트 SaaS 클라우드" value="saas" %}}
도메인 캡처를 사용하면 `@example.com` 같은 회사 이메일로 W&B SaaS 클라우드 조직에 자동으로 추가할 수 있습니다. 이로써 임직원이 올바른 조직에 합류하고, 신규 사용자가 회사 밖에 자산을 생성하는 걸 방지할 수 있습니다.

다음 표는 도메인 캡처 활성화 여부에 따른 신규 및 기존 사용자의 변화를 요약합니다.

| | 도메인 캡처 사용 | 도메인 캡처 미사용 |
| ----- | ----- | ----- |
| 신규 사용자 | 인증된 도메인에서 가입한 사용자는 기본 팀의 멤버로 자동 추가됩니다. 팀 가입을 활성화할 경우, 가입 시 추가로 팀을 선택할 수 있습니다. 초대장이 있으면 다른 조직/팀에도 가입 가능합니다. | 중앙 조직이 있다는 사실을 모른 채 W&B 계정을 별도로 만들 수 있습니다. | 
| 초대된 사용자 | 초대 수락 시 자동으로 조직에 합류합니다. 단, 이들은 기본 팀에는 자동 추가되지 않습니다. 초대장이 있으면 다른 조직/팀에도 가입 가능합니다. | 초대 수락 시 자동으로 조직에 합류합니다. 초대장이 있으면 다른 조직/팀에도 가입 가능합니다. | 
| 기존 사용자 | 조직 도메인 이메일을 가진 기존 사용자는 W&B 앱 내에서 좋은 팀에 가입할 수 있습니다. 기존 사용자가 만든 데이터는 그대로 남습니다. 데이터 마이그레이션은 없습니다. | 기존 사용자는 여러 조직/팀에 분산되어 있을 수 있습니다. |

조직에 신규 초대 없는 사용자를 기본 팀에 자동 배정하려면:

1. https://wandb.ai/home 으로 이동합니다.
2. 우측 상단 **User menu** 드롭다운에서 **Settings**를 선택합니다.
3. **Settings** 탭에서 **General**을 선택하세요.
4. **Domain capture** 아래 **Claim domain** 버튼을 클릭합니다.
5. **Default team** 드롭다운에서 신규 사용자가 자동 참여할 팀을 선택하세요. 팀이 보이지 않는 경우 Teams 설정을 업데이트해야 합니다. 자세한 내용은 [Add and manage teams]({{< relref path="#add-and-manage-teams" lang="ko" >}}) 를 참고하세요.
6. **Claim email domain** 버튼을 클릭합니다.

신규 초대 없는 사용자를 팀에 자동 할당하려면, 해당 팀의 설정에서 도메인 매칭을 먼저 활성화해야 합니다.

1. `https://wandb.ai/<team-name>`의 대시보드로 이동합니다. `<team-name>` 부분을 원하는 팀명으로 대체하세요.
2. 왼쪽 네비게이션에서 **Team settings**를 선택하세요.
3. **Privacy** 섹션에서 "Recommend new users with matching email domains join this team upon signing up" 옵션을 토글하세요.

{{% /tab %}}
{{% tab header="전용 또는 셀프 매니지드" value="dedicated" %}}
Dedicated 또는 Self-managed 배포를 사용하는 경우 도메인 캡처 구성을 위해 W&B Account Team에 문의하세요. 설정이 완료되면, W&B SaaS 인스턴스는 회사 이메일 주소로 W&B 계정을 생성하는 사용자에게 관리자의 연락처 정보를 안내하여 전용 또는 셀프 매니지드 인스턴스에 엑세스를 요청하도록 안내합니다.

| | 도메인 캡처 사용 | 도메인 캡처 미사용 |
| ----- | ----- | -----|
| 신규 사용자 | SaaS 클라우드에서 인증된 도메인으로 가입한 사용자는 관리자가 설정한 이메일 주소로 엑세스 요청 연락을 하도록 자동 안내됩니다. 그래도 SaaS 클라우드에서 Trial 용 조직을 생성할 수 있습니다. | 중앙 전용 인스턴스가 있는지 모른 채 W&B SaaS 계정을 생성할 수 있습니다. | 
| 기존 사용자 | 기존 사용자는 여러 조직/팀에 분산되어 있을 수 있습니다.| 기존 사용자는 여러 조직/팀에 분산되어 있을 수 있습니다.|
{{% /tab %}}
{{< /tabpane >}}


### 사용자 역할 할당 또는 수정

조직 내 모든 멤버는 W&B Models 와 Weave 각각의 조직 역할과 좌석을 보유합니다. 각 사용자의 좌석 유형은 요금 및 각 제품에서 수행할 수 있는 작업 범위에 영향을 미칩니다.

사용자를 조직에 초대할 때 역할을 부여하며, 언제든 해당 역할을 변경할 수 있습니다.

조직 내 사용자는 아래 중 한 가지 역할을 가질 수 있습니다.

| 역할 | 설명 |
| ----- | ----- |
| admin| 조직에 다른 사용자를 추가/삭제하고, 사용자 역할/커스텀 역할을 관리하며, 팀도 추가할 수 있는 인스턴스 관리자입니다. 관리자가 부재시를 대비해 2명 이상의 관리자를 두는 것을 권장합니다. |
| Member | 인스턴스 관리자가 초대한 일반 사용자입니다. 조직의 member는 다른 사용자 초대나 관리 권한이 없습니다. |
| Viewer (Enterprise 전용) | 인스턴스 관리자가 초대한 읽기 전용 사용자입니다. 오직 조직과 소속 팀에 대한 읽기 권한만 있습니다. |
| Custom Roles (Enterprise 전용) | 조직 관리자가 앞서 언급한 View-Only 또는 Member 기반으로 세분화된 권한을 갖는 새 역할을 직접 생성할 수 있습니다. 팀 관리자는 자신 팀의 사용자에게 커스텀 역할을 할당할 수 있습니다.|

사용자 역할을 변경하려면:

1. https://wandb.ai/home 으로 이동합니다.
2. 우측 상단의 **User menu** 드롭다운에서 **Users**를 선택합니다.
4. 검색창에 사용자 이름 또는 이메일을 입력하세요.
4. 사용자 이름 옆의 **TEAM ROLE** 드롭다운에서 역할을 선택하세요.

### 사용자의 엑세스 권한 할당 또는 변경

조직 내 사용자는 모델 좌석 또는 weave 엑세스 유형(Full, Viewer, No access) 중 하나를 가질 수 있습니다.

| 좌석 유형 | 설명 |
| ----- | ----- |
| Full | 이 역할은 Models 또는 Weave 데이터에 쓰기, 읽기, 내보내기 등 모든 권한이 있습니다. |
| Viewer | 조직과 소속 팀에 대한 읽기만 가능하며, Models 또는 Weave 역시 읽기 전용입니다. |
| No access | Models 또는 Weave 제품에 엑세스할 수 없습니다. |

모델 좌석 및 weave 엑세스 유형은 조직 레벨에서 정의되며, 팀에 상속됩니다. 사용자의 좌석 유형을 변경하려면, 조직 설정에서 아래 단계를 따르세요.

1. SaaS 사용자는 `https://wandb.ai/account-settings/<organization>/settings`에서 조직 설정에 접근하세요. 각괄호(`< >`) 부분을 조직명으로 교체하세요. Dedicated 및 Self-managed 배포의 경우는 `https://<your-instance>.wandb.io/org/dashboard`로 이동하세요.
2. **Users** 탭을 선택합니다.
3. **Role** 드롭다운에서 해당 사용자에게 할당할 좌석 유형을 선택합니다.

{{% alert %}}
조직 역할 및 구독 유형에 따라 조직 내에서 가용한 좌석 유형이 달라집니다.
{{% /alert %}}

### 사용자 제거

1. https://wandb.ai/home 으로 이동합니다.
2. 우측 상단 **User menu** 드롭다운의 **Users**를 선택합니다.
4. 검색창에 이름 또는 이메일을 입력하세요.
5. 해당 사용자가 보이면 점 세 개 아이콘(**...**)을 클릭하세요.
6. **Remove member**를 선택하세요.

### 결제 관리자 지정
1. https://wandb.ai/home 으로 이동합니다.
2. 우측 상단 **User menu** 드롭다운의 **Users**를 선택합니다.
4. 검색창에 이름 또는 이메일을 입력하세요.
5. **Billing admin** 컬럼에서 결제 관리자로 지정할 사용자를 선택하세요.


## 팀 추가 및 관리
조직의 대시보드를 사용하여 Teams를 생성하고 관리할 수 있습니다. 조직 관리자 또는 팀 관리자는 다음을 할 수 있습니다:
- 팀에 사용자 초대 또는 제거
- 팀 멤버의 역할 관리
- 신규 사용자가 조직에 가입할 때 자동으로 팀에 추가
- 팀 대시보드를 통해 팀 저장소 관리 (`https://wandb.ai/<team-name>`)

### 팀 생성

조직 대시보드를 이용해 팀을 만드세요:

1. https://wandb.ai/home 으로 이동합니다.
2. 왼쪽 네비게이션 패널의 **Teams** 아래에 위치한 **Create a team to collaborate** 를 선택합니다.
{{< img src="/images/hosting/create_new_team.png" alt="Create new team" >}}
3. 나타나는 모달의 **Team name** 입력란에 팀 이름을 적으세요.
4. 저장소 타입을 선택합니다.
5. **Create team** 버튼을 클릭합니다.

**Create team** 버튼을 클릭하면 W&B가 `https://wandb.ai/<team-name>` 페이지로 이동시켜줍니다. `<team-name>`에는 팀 생성 때 입력한 이름이 표시됩니다.

팀 생성 후에는 팀에 사용자를 추가할 수 있습니다.

### 팀에 사용자 초대

소속 팀 대시보드를 통해, 이메일 또는 이미 W&B 계정을 가진 경우 W&B 사용자명으로 팀에 초대할 수 있습니다.

1. `https://wandb.ai/<team-name>`으로 이동합니다.
2. 왼쪽 네비게이션에서 **Team settings**을 선택합니다.
{{< img src="/images/hosting/team_settings.png" alt="Team settings" >}}
3. **Users** 탭을 선택합니다.
4. **Invite a new user**를 클릭하세요.
5. 나타나는 모달의 **Email or username** 입력란에 사용자 이메일을 입력하고, **Select a team** 역할 드롭다운에서 해당 사용자에게 할당할 팀 역할을 선택하세요. 팀 사용자의 역할에 관한 자세한 내용은 [Team roles]({{< relref path="#assign-or-update-a-team-members-role" lang="ko" >}})을 참고하세요.
6. **Send invite** 버튼을 클릭합니다.

기본적으로 팀 또는 인스턴스 관리자만 팀 멤버 초대가 가능합니다. 이 설정을 변경하려면 [Team settings]({{< relref path="/guides/models/app/settings-page/team-settings.md#privacy" lang="ko" >}})를 참고하세요.

이메일 초대 외에도, [조직의 도메인 캡처]({{< relref path="#domain-capture" lang="ko" >}}) 를 통해 신규 사용자를 자동으로 팀에 추가할 수 있습니다.

### 회원 가입 시 팀 조직과 매칭

조직 내 신규 사용자가 W&B 계정 가입 시, Verified Email Domain 이 조직의 Verified 도메인과 일치하면, 조직에 속한 Teams를 목록 형태로 볼 수 있습니다. (조직 관리자가 도메인 클레임을 미리 활성화해야 함)

도메인 캡처 활성화 절차는 [Domain capture]({{< relref path="#domain-capture" lang="ko" >}})를 참고하세요.


### 팀 멤버의 역할 부여 또는 수정


1. 팀 멤버 이름 옆의 계정 타입 아이콘을 클릭하세요.
2. 드롭다운에서 멤버에게 부여할 계정 유형을 선택하세요.

팀에 할당할 수 있는 역할은 다음과 같습니다.

| 역할   |   정의   |
|-----------|---------------------------|
| admin    | 팀 내 사용자 추가/제거, 역할 변경, 팀 설정을 구성할 수 있는 사용자 |
| Member    | 이메일 또는 조직-level 사용자명으로 초대된 일반 팀 사용자. 다른 사용자를 팀에 초대할 수 없습니다.  |
| View-Only (Enterprise 한정) | 이메일 또는 조직-level 사용자명으로 초대된 팀 읽기 전용 사용자. 팀 및 그 소속 데이터에 대해서만 읽기 가능합니다.  |
| Service (Enterprise 한정)   | 서비스 워커(계정)는 W&B run 자동화 도구와 함께 사용할 때 유용한 API 키입니다. 팀의 서비스 계정 API 키를 사용할 경우, run attribution을 위해 반드시 환경변수 `WANDB_USERNAME` 을 지정하세요. |
| Custom Roles (Enterprise 한정)   | 조직 관리자는 View-Only 또는 Member 역할을 기반으로 권한을 세분화한 커스텀 역할을 생성할 수 있습니다. 팀 관리자는 각 팀 사용자에게 커스텀 역할을 할당할 수 있습니다. 자세한 내용은 [커스텀 역할 안내](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3)를 참고하세요. |

{{% alert %}}
전용 클라우드 또는 Self-managed 배포에서만 팀 멤버에게 커스텀 역할을 할당할 수 있습니다. (Enterprise 라이선스 필요)
{{% /alert %}}

### 팀에서 사용자 제거
팀 대시보드를 통해 사용자를 제거할 수 있습니다. W&B에서는, 팀 멤버가 탈퇴해도 해당 멤버가 생성한 run은 팀에 남아있게 됩니다.

1. `https://wandb.ai/<team-name>`으로 이동합니다.
2. 왼쪽 네비게이션 바에서 **Team settings**을 선택합니다.
3. **Users** 탭을 선택하세요.
4. 제거하고 싶은 사용자의 이름 옆에 마우스를 올려 둡니다. 점 세 개 아이콘(**...**)이 보이면 클릭하세요.
5. 드롭다운에서 **Remove user**를 선택합니다.