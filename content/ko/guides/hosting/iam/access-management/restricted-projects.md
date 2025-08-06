---
title: 프로젝트 엑세스 제어 관리
description: 프로젝트 엑세스는 공개 범위와 프로젝트 수준 역할을 사용해 관리하세요
menu:
  default:
    identifier: ko-guides-hosting-iam-access-management-restricted-projects
    parent: access-management
---

W&B 프로젝트의 범위를 정의하여 누가 해당 프로젝트를 보고, 수정하고, W&B run 을 제출할 수 있는지 제한할 수 있습니다.

W&B 팀 내의 어떤 프로젝트든 접근 수준을 지정하기 위해 여러 가지 제어 옵션을 조합해 사용할 수 있습니다. **가시성 범위(Visibility scope)** 가 상위 단계의 메커니즘입니다. 이를 사용해서 어떤 사용자 그룹이 프로젝트를 보고 run 을 제출할 수 있는지 제어할 수 있습니다. _Team_ 또는 _Restricted_ 가시성 범위를 가진 프로젝트의 경우, **프로젝트 수준 역할(Project level roles)** 을 사용하여 각 사용자가 프로젝트 내에서 갖는 엑세스 수준을 제어할 수 있습니다.

{{% alert %}}
프로젝트 소유자, 팀 관리자, 또는 조직 관리자가 프로젝트의 가시성을 설정하거나 수정할 수 있습니다.
{{% /alert %}}

## 가시성 범위

총 네 가지 프로젝트 가시성 범위를 선택할 수 있습니다. 공개 정도가 높은 순서대로 아래와 같습니다.

| 범위 | 설명 |
| ----- | ----- |
| Open |프로젝트에 대해 알고 있는 누구나 볼 수 있고 run 또는 report 를 제출할 수 있습니다.|
| Public |프로젝트에 대해 알고 있는 누구나 볼 수 있습니다. run 또는 report 제출은 팀만 할 수 있습니다.|
| Team |부모 팀의 팀 멤버만 프로젝트를 보고 run 이나 report 를 제출할 수 있습니다. 팀 외부 사용자는 프로젝트에 엑세스할 수 없습니다.|
| Restricted|부모 팀에서 초대한 멤버만 프로젝트를 보고 run 또는 report 를 제출할 수 있습니다.|

{{% alert %}}
민감하거나 기밀인 데이터와 관련된 워크플로우에서 협업하려면 프로젝트의 범위를 **Restricted** 로 설정하세요. 팀 내에서 Restricted 프로젝트를 만들 때, 해당 실험, Artifacts, Reports 등과 관련해 협업이 필요한 팀 멤버들을 초대하거나 추가할 수 있습니다.

다른 프로젝트 범위와 달리, 팀의 모든 멤버가 자동으로 restricted 프로젝트에 엑세스 권한을 가지지 않습니다. 단, 필요할 경우 팀 관리자는 restricted 프로젝트에 참여할 수 있습니다.
{{% /alert %}}

### 새 프로젝트 또는 기존 프로젝트에 가시성 범위 설정하기

프로젝트 생성 시 또는 추후 편집할 때 프로젝트의 가시성 범위를 설정할 수 있습니다.

{{% alert %}}
* 프로젝트 소유자 또는 팀 관리자만 프로젝트의 가시성 범위를 설정하거나 수정할 수 있습니다.
* 팀 관리자가 팀의 개인정보 설정에서 **Make all future team projects private (public sharing not allowed)** 옵션을 활성화하면, 해당 팀에서는 **Open** 및 **Public** 프로젝트 가시성 옵션이 비활성화됩니다. 이 경우, 팀은 **Team** 및 **Restricted** 범위만 사용할 수 있습니다.
{{% /alert %}}

#### 새 프로젝트 생성 시 가시성 범위 지정하기

1. SaaS Cloud, Dedicated Cloud 또는 셀프 관리 인스턴스에서 본인의 W&B 조직으로 이동하세요.
2. 왼쪽 사이드바의 **My projects** 섹션에서 **Create a new project** 버튼을 클릭하세요. 또는 팀의 **Projects** 탭으로 이동해서 오른쪽 상단의 **Create new project** 버튼을 클릭해도 됩니다.
3. 부모 팀을 선택하고 프로젝트 이름을 입력한 뒤, **Project Visibility** 드롭다운에서 원하는 범위를 선택하세요.
{{< img src="/images/hosting/restricted_project_add_new.gif" alt="Creating restricted project" >}}

**Restricted** 가시성을 선택한 경우, 다음 단계를 진행하세요.

4. **Invite team members** 필드에 W&B 팀 멤버 한 명 이상을 입력하세요. 프로젝트 협업에 꼭 필요한 멤버만 추가하는 것이 좋습니다.
{{< img src="/images/hosting/restricted_project_2.png" alt="Restricted project configuration" >}}

{{% alert %}}
추후 restricted 프로젝트의 **Users** 탭에서 멤버를 추가하거나 제거할 수 있습니다.
{{% /alert %}}

#### 기존 프로젝트의 가시성 범위 수정하기

1. W&B Project 로 이동하세요.
2. 왼쪽 컬럼에서 **Overview** 탭을 선택하세요.
3. 오른쪽 상단의 **Edit Project Details** 버튼을 클릭하세요.
4. **Project Visibility** 드롭다운에서 원하는 범위를 선택하세요.
{{< img src="/images/hosting/restricted_project_edit.gif" alt="Editing restricted project settings" >}}

**Restricted** 가시성을 선택한 경우, 다음 단계를 진행하세요.

5. 프로젝트 내 **Users** 탭으로 이동해서 **Add user** 버튼을 클릭하여 restricted 프로젝트에 초대할 사용자들을 지정하세요.

{{% alert color="secondary" %}}
* 프로젝트의 가시성 범위를 **Team** 에서 **Restricted** 로 변경하면, 필요한 팀 멤버를 초대하지 않는 한 팀의 모든 멤버가 해당 프로젝트에 대한 엑세스 권한을 잃게 됩니다.
* 프로젝트의 가시성 범위를 **Restricted** 에서 **Team** 으로 변경하면 팀의 모든 멤버가 프로젝트에 엑세스할 수 있습니다.
* restricted 프로젝트의 사용자 목록에서 팀 멤버를 제거하면, 해당 멤버는 프로젝트에 더 이상 엑세스할 수 없습니다.
{{% /alert %}}

### Restricted 범위의 주의사항

* restricted 프로젝트에서 팀 서비스 계정을 사용하려면, 해당 계정을 프로젝트에 직접 초대하거나 추가해야 합니다. 그렇지 않을 경우 기본적으로 restricted 프로젝트에 엑세스할 수 없습니다.
* restricted 프로젝트에서 run 을 다른 곳으로 이동할 수는 없지만, non-restricted 프로젝트에서 restricted 프로젝트로 run 을 이동하는 것은 가능합니다.
* restricted 프로젝트의 가시성은 팀 개인정보 설정 옵션인 **Make all future team projects private (public sharing not allowed)** 와 관계없이 언제든지 **Team** 범위로 전환할 수 있습니다.
* restricted 프로젝트의 소유자가 더 이상 부모 팀의 멤버가 아닌 경우, 팀 관리자가 소유자를 변경해야 프로젝트 운영에 차질이 없습니다.

## 프로젝트 수준 역할

팀 내에서 _Team_ 또는 _Restricted_ 범위의 프로젝트의 경우, 사용자의 팀 수준 역할과 다르게 사용자별로 프로젝트 수준의 역할을 지정할 수 있습니다. 예를 들어 한 사용자가 팀 수준에서 _Member_ 역할을 가지고 있더라도, 특정 _Team_ 또는 _Restricted_ 프로젝트에서는 _View-Only_, _Admin_, 혹은 적용 가능한 커스텀 역할을 지정할 수 있습니다.

{{% alert %}}
프로젝트 수준 역할은 SaaS Cloud, Dedicated Cloud 및 셀프 관리 인스턴스에서 프리뷰 기능입니다.
{{% /alert %}}

### 사용자에게 프로젝트 수준 역할 할당하기

1. W&B Project 로 이동하세요.
2. 왼쪽 컬럼의 **Overview** 탭을 선택하세요.
3. 프로젝트 내 **Users** 탭으로 이동하세요.
4. **Project Role** 필드에서 해당 사용자의 현재 역할을 클릭하면, 다른 사용 가능한 역할 목록이 나타납니다.
5. 드롭다운에서 원하는 역할을 선택하세요. 선택 즉시 저장됩니다.

{{% alert %}}
사용자의 프로젝트 수준 역할이 팀 수준 역할과 다를 경우, 프로젝트 수준 역할 옆에 **\*** 표시가 추가됩니다.
{{% /alert %}}

### 프로젝트 수준 역할의 주요 사항

* 기본적으로 _team_ 또는 _restricted_ 범위의 프로젝트에 참여하는 모든 사용자는 자신의 팀 수준 역할을 **상속** 받습니다.
* 팀 수준에서 _View-only_ 역할을 가진 사용자의 프로젝트 수준 역할은 변경할 수 없습니다.
* 특정 프로젝트에서 사용자의 프로젝트 역할이 팀 수준 역할과 **동일** 하다면, 추후 팀 관리자가 팀 역할을 변경할 경우 해당 프로젝트 역할도 자동으로 반영되어 따라갑니다.
* 특정 프로젝트에서 사용자의 프로젝트 역할이 팀 수준 역할과 **다르** 게 설정된 경우, 나중에 팀 관리자가 팀 역할을 변경해도 해당 프로젝트의 역할은 그대로 유지됩니다.
* _restricted_ 프로젝트에서 사용자 역할이 팀 역할과 달랐다가 해당 사용자를 프로젝트에서 제거했다가 다시 추가하는 경우, 기본 동작에 따라 팀 역할이 상속됩니다. 이때 필요하다면 다시 프로젝트 수준 역할을 팀 역할과 다르게 직접 지정해주어야 합니다.