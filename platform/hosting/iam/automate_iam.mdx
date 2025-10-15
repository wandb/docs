---
title: Automate user and team management
menu:
  default:
    identifier: ko-guides-hosting-iam-automate_iam
    parent: identity-and-access-management-iam
weight: 3
---

## SCIM API

SCIM API를 사용하여 사용자 및 사용자가 속한 Teams를 효율적이고 반복 가능한 방식으로 관리합니다. SCIM API를 사용하여 사용자 정의 역할을 관리하거나 W&B organization의 사용자에게 역할을 할당할 수도 있습니다. 역할 엔드포인트는 공식 SCIM 스키마의 일부가 아닙니다. W&B는 사용자 정의 역할의 자동 관리를 지원하기 위해 역할 엔드포인트를 추가합니다.

SCIM API는 다음과 같은 경우에 특히 유용합니다.

* 사용자 프로비저닝 및 프로비저닝 해제를 대규모로 관리하려는 경우
* SCIM을 지원하는 ID 공급자로 사용자를 관리하려는 경우

SCIM API는 크게 **User**, **Group**, **Roles**의 세 가지 범주로 나뉩니다.

### User SCIM API

[User SCIM API]({{< relref path="./scim.md#user-resource" lang="ko" >}})를 사용하면 W&B organization에서 사용자를 생성, 비활성화, 사용자 세부 정보를 가져오거나 모든 사용자 목록을 가져올 수 있습니다. 이 API는 organization의 사용자에게 미리 정의된 역할 또는 사용자 정의 역할을 할당하는 기능도 지원합니다.

{{% alert %}}
`DELETE User` 엔드포인트를 사용하여 W&B organization 내에서 사용자를 비활성화합니다. 비활성화된 사용자는 더 이상 로그인할 수 없습니다. 그러나 비활성화된 사용자는 organization의 사용자 목록에 계속 표시됩니다.

비활성화된 사용자를 사용자 목록에서 완전히 제거하려면 [organization에서 사용자를 제거]({{< relref path="access-management/manage-organization.md#remove-a-user" lang="ko" >}})해야 합니다.

필요한 경우 비활성화된 사용자를 다시 활성화할 수 있습니다.
{{% /alert %}}

### Group SCIM API

[Group SCIM API]({{< relref path="./scim.md#group-resource" lang="ko" >}})를 사용하면 organization에서 Teams를 생성하거나 제거하는 것을 포함하여 W&B Teams를 관리할 수 있습니다. `PATCH Group`을 사용하여 기존 Team에서 사용자를 추가하거나 제거합니다.

{{% alert %}}
W&B에는 `동일한 역할을 가진 사용자 그룹`이라는 개념이 없습니다. W&B Team은 그룹과 매우 유사하며, 서로 다른 역할을 가진 다양한 사용자가 관련 Projects 집합에서 공동으로 작업할 수 있도록 합니다. Teams는 서로 다른 사용자 그룹으로 구성될 수 있습니다. Team의 각 사용자에게 Team 관리자, 멤버, 뷰어 또는 사용자 정의 역할과 같은 역할을 할당합니다.

W&B는 그룹과 W&B Teams 간의 유사성 때문에 Group SCIM API 엔드포인트를 W&B Teams에 매핑합니다.
{{% /alert %}}

### Custom role API

[Custom role SCIM API]({{< relref path="./scim.md#role-resource" lang="ko" >}})를 사용하면 organization에서 사용자 정의 역할을 생성, 나열 또는 업데이트하는 것을 포함하여 사용자 정의 역할을 관리할 수 있습니다.

{{% alert color="secondary" %}}
사용자 정의 역할을 삭제할 때는 주의하십시오.

`DELETE Role` 엔드포인트를 사용하여 W&B organization 내에서 사용자 정의 역할을 삭제합니다. 사용자 정의 역할이 상속하는 미리 정의된 역할은 작업 전에 사용자 정의 역할이 할당된 모든 사용자에게 할당됩니다.

`PUT Role` 엔드포인트를 사용하여 사용자 정의 역할에 대해 상속된 역할을 업데이트합니다. 이 작업은 사용자 정의 역할의 기존 사용자 정의 권한(즉, 상속되지 않은 사용자 정의 권한)에는 영향을 주지 않습니다.
{{% /alert %}}

## W&B Python SDK API

SCIM API를 통해 사용자 및 Team 관리를 자동화할 수 있는 것처럼 [W&B Python SDK API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})에서 사용할 수 있는 일부 메소드를 사용하여 이 목적을 달성할 수도 있습니다. 다음 메소드를 기록해 두십시오.

| Method name | Purpose |
|-------------|---------|
| `create_user(email, admin=False)` | organization에 사용자를 추가하고 선택적으로 organization 관리자로 만듭니다. |
| `user(userNameOrEmail)` | organization에서 기존 사용자를 반환합니다. |
| `user.teams()` | 사용자의 Teams를 반환합니다. user(userNameOrEmail) 메소드를 사용하여 사용자 오브젝트를 가져올 수 있습니다. |
| `create_team(teamName, adminUserName)` | 새 Team을 만들고 선택적으로 organization 수준의 사용자를 Team 관리자로 만듭니다. |
| `team(teamName)` | organization에서 기존 Team을 반환합니다. |
| `Team.invite(userNameOrEmail, admin=False)` | Team에 사용자를 추가합니다. team(teamName) 메소드를 사용하여 Team 오브젝트를 가져올 수 있습니다. |
| `Team.create_service_account(description)` | Team에 서비스 계정을 추가합니다. team(teamName) 메소드를 사용하여 Team 오브젝트를 가져올 수 있습니다. |
|` Member.delete()` | Team에서 멤버 사용자를 제거합니다. team 오브젝트의 `members` 속성을 사용하여 Team에서 멤버 오브젝트 목록을 가져올 수 있습니다. 그리고 team(teamName) 메소드를 사용하여 Team 오브젝트를 가져올 수 있습니다. |
