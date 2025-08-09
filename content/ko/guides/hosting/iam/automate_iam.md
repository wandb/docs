---
title: 사용자 및 팀 관리 자동화
menu:
  default:
    identifier: ko-guides-hosting-iam-automate_iam
    parent: identity-and-access-management-iam
weight: 3
---

## SCIM API

SCIM API를 사용하면 사용자와 그들이 속한 팀을 효율적이고 반복적으로 관리할 수 있습니다. 또한 SCIM API를 활용하여 커스텀 역할을 관리하거나, W&B 조직 내 사용자에게 역할을 할당할 수 있습니다. 역할 관련 엔드포인트는 공식 SCIM 스키마에 속해 있지 않으며, W&B에서 자동화된 커스텀 역할 관리를 지원하기 위해 별도로 제공합니다.

SCIM API는 다음과 같은 경우에 특히 유용합니다:

* 대규모로 사용자 프로비저닝 및 해제를 관리할 때
* SCIM을 지원하는 아이덴티티 프로바이더로 사용자를 관리할 때

SCIM API는 크게 **User**, **Group**, **Roles** 세 가지 범주로 나눌 수 있습니다.

### User SCIM API

[User SCIM API]({{< relref path="./scim.md#user-resource" lang="ko" >}})를 사용하면 W&B 조직 내에서 사용자를 생성, 비활성화, 세부 정보 조회 또는 모든 사용자 목록을 확인할 수 있습니다. 또한 이 API는 조직 내 사용자에게 미리 정의된 역할이나 커스텀 역할을 할당하는 것도 지원합니다.

{{% alert %}}
W&B 조직 내에서 사용자를 비활성화하려면 `DELETE User` 엔드포인트를 사용하세요. 비활성화된 사용자는 더 이상 로그인할 수 없습니다. 하지만 비활성화된 사용자는 여전히 조직의 사용자 목록에는 나타납니다.

비활성화된 사용자를 사용자 목록에서 완전히 제거하려면, 반드시 [조직에서 사용자를 삭제]({{< relref path="access-management/manage-organization.md#remove-a-user" lang="ko" >}})해야 합니다.

필요하다면, 비활성화된 사용자를 다시 활성화하는 것도 가능합니다.
{{% /alert %}}

### Group SCIM API

[Group SCIM API]({{< relref path="./scim.md#group-resource" lang="ko" >}})를 사용하면 W&B 팀(Teams)을 관리할 수 있으며, 조직 내에서 팀을 생성하거나 제거할 수 있습니다. 기존 팀에 사용자를 추가하거나 삭제하려면 `PATCH Group`을 사용하세요.

{{% alert %}}
W&B에는 `동일한 역할의 사용자가 모인 그룹`이라는 개념이 존재하지 않습니다. W&B의 팀은 그룹과 유사하지만, 서로 다른 역할을 지닌 다양한 인물이 연관된 프로젝트 집합에서 협업할 수 있도록 도와줍니다. 팀은 여러 서로 다른 사용자 그룹으로 구성될 수 있습니다. 각 팀의 사용자는 팀 관리자, 멤버, 뷰어, 혹은 커스텀 역할 중 하나를 할당받습니다.

W&B에서는 Group SCIM API 엔드포인트를 팀에 매핑하여, 그룹과 팀 개념의 유사성을 활용합니다.
{{% /alert %}}

### Custom role API

[Custom role SCIM API]({{< relref path="./scim.md#role-resource" lang="ko" >}})를 통해 조직 내 커스텀 역할을 생성, 목록 확인, 또는 업데이트할 수 있습니다.

{{% alert color="secondary" %}}
커스텀 역할을 삭제할 때는 신중하게 진행하세요.

W&B 조직 내에서 커스텀 역할을 삭제할 때는 `DELETE Role` 엔드포인트를 사용합니다. 이 작업 전에 해당 커스텀 역할을 할당받은 모든 사용자에게, 커스텀 역할이 상속했던 미리 정의된 역할이 자동으로 할당됩니다.

`PUT Role` 엔드포인트로 커스텀 역할의 상속 역할을 변경할 수 있습니다. 이 작업은 해당 커스텀 역할 내에서 이미 존재하는(즉, 상속되지 않은) 커스텀 권한에는 영향을 주지 않습니다.
{{% /alert %}}

## W&B Python SDK API

SCIM API를 활용해 사용자와 팀을 자동화해 관리할 수 있는 것처럼, [W&B Python SDK API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})의 일부 메소드들도 이 목적에 사용할 수 있습니다. 아래 메소드들을 참고하세요.

| 메소드 이름 | 용도 |
|-------------|---------|
| `create_user(email, admin=False)` | 조직에 사용자를 추가하고, 필요 시 조직 관리자로 지정할 수 있습니다. |
| `user(userNameOrEmail)` | 조직에서 기존 사용자를 반환합니다. |
| `user.teams()` | 해당 사용자가 속한 팀 목록을 반환합니다. 사용자 오브젝트는 user(userNameOrEmail) 메소드로 얻을 수 있습니다. |
| `create_team(teamName, adminUserName)` | 새 팀을 만들고, 필요하다면 조직 내 사용자를 팀의 관리자(Team admin)로 지정할 수 있습니다. |
| `team(teamName)` | 조직 내 기존 팀을 반환합니다. |
| `Team.invite(userNameOrEmail, admin=False)` | 팀에 사용자를 추가합니다. 팀 오브젝트는 team(teamName) 메소드로 가져올 수 있습니다. |
| `Team.create_service_account(description)` | 팀에 서비스 계정을 추가합니다. 팀 오브젝트는 team(teamName) 메소드로 가져올 수 있습니다. |
|` Member.delete()` | 팀에서 멤버 사용자를 삭제합니다. 팀 오브젝트의 `members` 속성을 통하여 팀 안의 멤버 오브젝트 리스트를 얻고, 팀 오브젝트는 team(teamName) 메소드로 얻을 수 있습니다. |