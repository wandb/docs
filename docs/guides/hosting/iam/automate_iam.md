---
title: Automate user and team management
displayed_sidebar: default
---

## SCIM API

SCIM API를 사용하면 사용자 및 그들이 속한 팀을 효율적이고 반복 가능한 방식으로 관리할 수 있습니다. 또한 SCIM API를 사용하여 사용자 정의 역할을 관리하거나 W&B 조직 내 사용자에게 역할을 할당할 수 있습니다. 역할 엔드포인트는 공식 SCIM 스키마의 일부가 아닙니다. W&B는 사용자 정의 역할의 자동 관리를 지원하기 위해 역할 엔드포인트를 추가합니다.

SCIM API는 다음의 경우 특히 유용합니다:

* 대규모 사용자 프로비저닝 및 프로비저닝 해제를 관리하는 경우
* SCIM을 지원하는 Identity Provider를 사용하여 사용자를 관리하는 경우

SCIM API는 크게 세 가지 범주로 나뉩니다 - **User**, **Group**, **Roles**.

### User SCIM API

[User SCIM API](./scim.md#user-resource)는 W&B 조직 내에서 사용자의 생성, 비활성화, 세부 정보 조회 또는 사용자 목록 조회를 할 수 있습니다. 이 API는 또한 조직 내 사용자에게 사전에 정의된 역할 또는 사용자 정의 역할을 할당하는 것을 지원합니다.

:::안내
W&B 조직 내의 사용자를 `DELETE User` 엔드포인트로 비활성화할 수 있습니다. 비활성화된 사용자는 더 이상 로그인할 수 없습니다. 그러나 비활성화된 사용자는 여전히 조직의 사용자 목록에 나타납니다.

비활성화된 사용자를 사용자 목록에서 완전히 제거하려면, [조직에서 사용자를 제거](manage-users.md#remove-a-user)해야 합니다.

필요하다면 비활성화된 사용자를 다시 활성화할 수 있습니다.
:::

### Group SCIM API

[Group SCIM API](./scim.md#group-resource)는 조직 내에서 W&B 팀을 관리하는 데 사용되며, 이를 통해 팀을 생성하거나 제거할 수 있습니다. `PATCH Group`를 사용하여 기존 팀에 사용자를 추가하거나 제거할 수 있습니다.

:::안내
W&B에는 `같은 역할을 가진 사용자 그룹`의 개념이 없습니다. W&B 팀은 그룹과 유사하며, 서로 다른 역할을 가진 다양한 인물이 관련 프로젝트를 공동 작업할 수 있도록 합니다. 팀은 서로 다른 사용자 그룹으로 구성될 수 있습니다. 팀 내 각 사용자에게 역할을 할당하십시오: 팀 관리자, 멤버, 뷰어 또는 사용자 정의 역할.

W&B는 Group SCIM API 엔드포인트를 W&B 팀과 매핑하여 그룹과 W&B 팀의 유사성을 반영합니다.
:::

### Custom role API

[Custom role SCIM API](./scim.md#role-resource)는 사용자 정의 역할을 관리할 수 있으며, 조직 내 사용자 정의 역할을 생성, 목록화 또는 업데이트할 수 있습니다.

:::주의
사용자 정의 역할 삭제 시 주의가 필요합니다.

`DELETE Role` 엔드포인트로 W&B 조직 내 사용자 정의 역할을 삭제합니다. 사용자 정의 역할이 상속했던 사전 정의된 역할이 사용자 정의 역할이 할당된 모든 사용자에게 할당됩니다.

`PUT Role` 엔드포인트로 사용자 정의 역할에 대해 상속된 역할을 업데이트합니다. 이 작업은 기존의, 즉 상속받지 않은 사용자 정의 권한에 영향을 미치지 않습니다.
:::

## W&B Python SDK API

SCIM API를 사용하여 사용자 및 팀 관리를 자동화할 수 있는 것처럼, [W&B Python SDK API](../../../ref/python/public-api/api.md)에 제공된 메소드를 사용하여 이 목적을 수행할 수 있습니다. 다음 메소드를 참고하세요:

| 메소드 이름 | 목적 |
|-------------|---------|
| `create_user(email, admin=False)` | 조직에 사용자를 추가하고, 선택적으로 관리자로 만들 수 있습니다. |
| `user(userNameOrEmail)` | 조직 내 기존 사용자를 반환합니다. |
| `user.teams()` | 사용자의 팀 목록을 반환합니다. 사용자 오브젝트는 user(userNameOrEmail) 메소드로 얻을 수 있습니다. |
| `create_team(teamName, adminUserName)` | 새 팀을 생성하고, 선택적으로 조직 수준의 사용자를 팀 관리자로 지정할 수 있습니다. |
| `team(teamName)` | 조직의 기존 팀을 반환합니다. |
| `Team.invite(userNameOrEmail, admin=False)` | 팀에 사용자를 추가합니다. 팀 오브젝트는 team(teamName) 메소드로 얻을 수 있습니다. |
| `Team.create_service_account(description)` | 팀에 서비스 계정을 추가합니다. 팀 오브젝트는 team(teamName) 메소드로 얻을 수 있습니다. |
| `Member.delete()` | 팀에서 멤버 사용자를 제거합니다. 팀 내 멤버 오브젝트 목록은 팀 오브젝트의 `members` 속성을 사용하여 얻을 수 있습니다. 팀 오브젝트는 team(teamName) 메소드로 얻을 수 있습니다. |