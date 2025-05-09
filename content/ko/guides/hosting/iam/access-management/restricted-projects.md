---
title: Manage access control for projects
description: 가시성 범위와 프로젝트 수준 역할을 사용하여 프로젝트 엑세스 를 관리합니다.
menu:
  default:
    identifier: ko-guides-hosting-iam-access-management-restricted-projects
    parent: access-management
---

W&B 프로젝트의 범위를 정의하여 누가 해당 프로젝트를 보고, 편집하고, W&B run을 제출할 수 있는지 제한합니다.

W&B 팀 내의 모든 프로젝트에 대한 엑세스 수준을 구성하기 위해 몇 가지 제어 기능을 함께 사용할 수 있습니다. **가시성 범위**는 더 높은 수준의 메커니즘입니다. 이를 사용하여 어떤 사용자 그룹이 프로젝트에서 run을 보거나 제출할 수 있는지 제어합니다. _Team_ 또는 _Restricted_ 가시성 범위를 가진 프로젝트의 경우 **프로젝트 수준 역할**을 사용하여 각 사용자가 프로젝트 내에서 갖는 엑세스 수준을 제어할 수 있습니다.

{{% alert %}}
프로젝트 소유자, 팀 관리자 또는 조직 관리자는 프로젝트의 가시성을 설정하거나 편집할 수 있습니다.
{{% /alert %}}

## 가시성 범위

선택할 수 있는 프로젝트 가시성 범위는 네 가지가 있습니다. 가장 공개적인 것부터 가장 사적인 것 순으로 나열하면 다음과 같습니다.

| 범위 | 설명 |
| ----- | ----- |
| Open | 프로젝트에 대해 알고 있는 사람은 누구나 보고 run 또는 리포트를 제출할 수 있습니다. |
| Public | 프로젝트에 대해 알고 있는 사람은 누구나 볼 수 있습니다. 팀만 run 또는 리포트를 제출할 수 있습니다. |
| Team | 상위 팀의 팀 멤버만 프로젝트를 보고 run 또는 리포트를 제출할 수 있습니다. 팀 외부의 사람은 프로젝트에 엑세스할 수 없습니다. |
| Restricted | 상위 팀에서 초대받은 팀 멤버만 프로젝트를 보고 run 또는 리포트를 제출할 수 있습니다. |

{{% alert %}}
민감하거나 기밀 데이터와 관련된 워크플로우에서 협업하려면 프로젝트의 범위를 **Restricted**로 설정하세요. 팀 내에서 제한된 프로젝트를 생성할 때 팀의 특정 팀 멤버를 초대하거나 추가하여 관련 Experiments, Artifacts, 리포트 등에서 협업할 수 있습니다.

다른 프로젝트 범위와 달리 팀의 모든 팀 멤버가 제한된 프로젝트에 대한 암묵적인 엑세스 권한을 얻는 것은 아닙니다. 동시에 팀 관리자는 필요한 경우 제한된 프로젝트에 참여할 수 있습니다.
{{% /alert %}}

### 새 프로젝트 또는 기존 프로젝트에서 가시성 범위 설정

프로젝트를 생성할 때 또는 나중에 편집할 때 프로젝트의 가시성 범위를 설정합니다.

{{% alert %}}
* 프로젝트 소유자 또는 팀 관리자만 가시성 범위를 설정하거나 편집할 수 있습니다.
* 팀 관리자가 팀의 개인 정보 설정 내에서 **향후 모든 팀 프로젝트를 비공개로 설정(공개 공유 불가)**를 활성화하면 해당 팀에 대해 **Open** 및 **Public** 프로젝트 가시성 범위가 해제됩니다. 이 경우 팀은 **Team** 및 **Restricted** 범위만 사용할 수 있습니다.
{{% /alert %}}

#### 새 프로젝트를 생성할 때 가시성 범위 설정

1. SaaS Cloud, 전용 클라우드 또는 자체 관리 인스턴스에서 W&B 조직으로 이동합니다.
2. 왼쪽 사이드바의 **내 프로젝트** 섹션에서 **새 프로젝트 만들기** 버튼을 클릭합니다. 또는 팀의 **Projects** 탭으로 이동하여 오른쪽 상단 모서리에 있는 **새 프로젝트 만들기** 버튼을 클릭합니다.
3. 상위 팀을 선택하고 프로젝트 이름을 입력한 후 **프로젝트 가시성** 드롭다운에서 원하는 범위를 선택합니다.
{{< img src="/images/hosting/restricted_project_add_new.gif" alt="" >}}

**Restricted** 가시성을 선택한 경우 다음 단계를 완료하십시오.

4. **팀 멤버 초대** 필드에 하나 이상의 W&B 팀 멤버 이름을 입력합니다. 프로젝트에서 협업하는 데 필수적인 팀 멤버만 추가하십시오.
{{< img src="/images/hosting/restricted_project_2.png" alt="" >}}

{{% alert %}}
**Users** 탭에서 나중에 제한된 프로젝트에서 팀 멤버를 추가하거나 제거할 수 있습니다.
{{% /alert %}}

#### 기존 프로젝트의 가시성 범위 편집

1. W&B 프로젝트로 이동합니다.
2. 왼쪽 열에서 **Overview** 탭을 선택합니다.
3. 오른쪽 상단 모서리에 있는 **프로젝트 세부 정보 편집** 버튼을 클릭합니다.
4. **프로젝트 가시성** 드롭다운에서 원하는 범위를 선택합니다.
{{< img src="/images/hosting/restricted_project_edit.gif" alt="" >}}

**Restricted** 가시성을 선택한 경우 다음 단계를 완료하십시오.

5. 프로젝트의 **Users** 탭으로 이동하여 **사용자 추가** 버튼을 클릭하여 특정 사용자를 제한된 프로젝트에 초대합니다.

{{% alert color="secondary" %}}
* 필요한 팀 멤버를 프로젝트에 초대하지 않으면 가시성 범위를 **Team**에서 **Restricted**로 변경하면 팀의 모든 팀 멤버가 프로젝트에 대한 엑세스 권한을 잃게 됩니다.
* 가시성 범위를 **Restricted**에서 **Team**으로 변경하면 팀의 모든 팀 멤버가 프로젝트에 대한 엑세스 권한을 얻게 됩니다.
* 제한된 프로젝트의 사용자 목록에서 팀 멤버를 제거하면 해당 프로젝트에 대한 엑세스 권한을 잃게 됩니다.
{{% /alert %}}

### 제한된 범위에 대한 기타 주요 참고 사항

* 제한된 프로젝트에서 팀 수준 서비스 계정을 사용하려면 해당 계정을 프로젝트에 특별히 초대하거나 추가해야 합니다. 그렇지 않으면 팀 수준 서비스 계정은 기본적으로 제한된 프로젝트에 엑세스할 수 없습니다.
* 제한된 프로젝트에서 run을 이동할 수는 없지만 제한되지 않은 프로젝트에서 제한된 프로젝트로 run을 이동할 수 있습니다.
* 팀 개인 정보 설정 **향후 모든 팀 프로젝트를 비공개로 설정(공개 공유 불가)**에 관계없이 제한된 프로젝트의 가시성을 **Team** 범위로만 변환할 수 있습니다.
* 제한된 프로젝트의 소유자가 더 이상 상위 팀에 속하지 않으면 팀 관리자는 프로젝트에서 원활한 운영을 보장하기 위해 소유자를 변경해야 합니다.

## 프로젝트 수준 역할

팀의 _Team_ 또는 _Restricted_ 범위 프로젝트의 경우 사용자에게 특정 역할을 할당할 수 있으며, 이는 해당 사용자의 팀 수준 역할과 다를 수 있습니다. 예를 들어 사용자가 팀 수준에서 _Member_ 역할을 하는 경우 해당 팀의 _Team_ 또는 _Restricted_ 범위 프로젝트 내에서 해당 사용자에게 _View-Only_ 또는 _Admin_ 또는 사용 가능한 사용자 지정 역할을 할당할 수 있습니다.

{{% alert %}}
프로젝트 수준 역할은 SaaS Cloud, 전용 클라우드 및 자체 관리 인스턴스에서 미리 보기로 제공됩니다.
{{% /alert %}}

### 사용자에게 프로젝트 수준 역할 할당

1. W&B 프로젝트로 이동합니다.
2. 왼쪽 열에서 **Overview** 탭을 선택합니다.
3. 프로젝트의 **Users** 탭으로 이동합니다.
4. **프로젝트 역할** 필드에서 해당 사용자에 대해 현재 할당된 역할을 클릭하면 다른 사용 가능한 역할 목록이 있는 드롭다운이 열립니다.
5. 드롭다운에서 다른 역할을 선택합니다. 즉시 저장됩니다.

{{% alert %}}
사용자의 프로젝트 수준 역할을 팀 수준 역할과 다르게 변경하면 프로젝트 수준 역할에 차이를 나타내는 **\***가 포함됩니다.
{{% /alert %}}

### 프로젝트 수준 역할에 대한 기타 주요 참고 사항

* 기본적으로 _team_ 또는 _restricted_ 범위 프로젝트의 모든 사용자에 대한 프로젝트 수준 역할은 해당 팀 수준 역할을 **상속**합니다.
* 팀 수준에서 _View-only_ 역할을 가진 사용자의 프로젝트 수준 역할은 **변경할 수 없습니다**.
* 특정 프로젝트 내에서 사용자의 프로젝트 수준 역할이 팀 수준 역할과 **동일**하고 팀 관리자가 팀 수준 역할을 변경하면 관련 프로젝트 역할이 자동으로 변경되어 팀 수준 역할을 추적합니다.
* 특정 프로젝트 내에서 사용자의 프로젝트 수준 역할을 팀 수준 역할과 **다르게** 변경하고 팀 관리자가 팀 수준 역할을 변경하면 관련 프로젝트 수준 역할은 그대로 유지됩니다.
* 프로젝트 수준 역할이 팀 수준 역할과 다른 경우 _restricted_ 프로젝트에서 사용자를 제거하고 일정 시간이 지난 후 사용자를 프로젝트에 다시 추가하면 기본 동작으로 인해 팀 수준 역할을 상속합니다. 필요한 경우 프로젝트 수준 역할을 다시 변경하여 팀 수준 역할과 다르게 만들어야 합니다.
