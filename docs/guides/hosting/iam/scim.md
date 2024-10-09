---
title: Manage users, groups, and roles with SCIM
displayed_sidebar: default
---

System for Cross-domain Identity Management (SCIM) API는 인스턴스나 조직의 관리자들이 그들의 W&B 조직에서 사용자, 그룹, 커스텀 역할을 관리할 수 있도록 합니다. SCIM 그룹은 W&B 팀에 매핑됩니다.

SCIM API는 `<host-url>/scim/`에서 엑세스 가능하며, [RC7643 프로토콜](https://www.rfc-editor.org/rfc/rfc7643)에서 찾을 수 있는 필드의 서브셋을 지원하는 `/Users`와 `/Groups` 엔드포인트를 제공하며, SCIM 공식 스키마의 일부가 아닌 `/Roles` 엔드포인트도 포함합니다. W&B는 W&B 조직에서 커스텀 역할의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가했습니다.

:::info
SCIM API는 [전용 클라우드](../hosting-options/dedicated_cloud.md), [셀프 관리 인스턴스](../hosting-options/self-managed.md), 및 [SaaS클라우드](../hosting-options/saas_cloud.md)를 포함한 모든 호스팅 옵션에 적용됩니다. SaaS 클라우드에서는 SCIM API 요청이 올바른 조직으로 가기 위해 조직 관리자가 사용자 설정에서 기본 조직을 설정해야 합니다. 이 설정은 사용자 설정의 `SCIM API Organization` 섹션에서 사용할 수 있습니다.
:::

## 인증

SCIM API는 API 키가 있는 기본 인증을 사용하여 인스턴스 또는 조직 관리자에 의해 엑세스할 수 있습니다. 기본 인증을 사용하여, `Authorization` 헤더에 `Basic`이라는 단어 뒤에 공백을 두고 `username:password`의 베이스64 인코딩된 문자열을 포함하여 HTTP 요청을 보냅니다. 여기서 `password`는 당신의 API 키입니다. 예를 들어, `demo:p@55w0rd`로 인증하려면 헤더는 `Authorization: Basic ZGVtbzpwQDU1dzByZA==`가 되어야 합니다.

## User 리소스

SCIM 사용자 리소스는 W&B 사용자와 매핑됩니다.

### 사용자 가져오기

- **엔드포인트:** **`<host-url>/scim/Users/{id}`**
- **메소드**: GET
- **설명**: 사용자의 고유 ID를 제공하여 [SaaS 클라우드](../hosting-options/saas_cloud.md) 조직 또는 [전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [셀프 관리 인스턴스](../hosting-options/self-managed.md) 인스턴스에서 특정 사용자에 대한 정보를 검색합니다.
- **요청 예제**:

```bash
GET /scim/Users/abc
```

- **응답 예제**:

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1"
}
```

### 사용자 목록

- **엔드포인트:** **`<host-url>/scim/Users`**
- **메소드**: GET
- **설명**: [SaaS 클라우드](../hosting-options/saas_cloud.md) 조직 또는 [전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [셀프 관리 인스턴스](../hosting-options/self-managed.md) 인스턴스의 모든 사용자 목록을 검색합니다.
- **요청 예제**:

```bash
GET /scim/Users
```

- **응답 예제**:

```bash
(Status 200)
```

```json
{
    "Resources": [
        {
            "active": true,
            "displayName": "Dev User 1",
            "emails": {
                "Value": "dev-user1@test.com",
                "Display": "",
                "Type": "",
                "Primary": true
            },
            "id": "abc",
            "meta": {
                "resourceType": "User",
                "created": "2023-10-01T00:00:00Z",
                "lastModified": "2023-10-01T00:00:00Z",
                "location": "Users/abc"
            },
            "schemas": [
                "urn:ietf:params:scim:schemas:core:2.0:User"
            ],
            "userName": "dev-user1"
        }
    ],
    "itemsPerPage": 9999,
    "schemas": [
        "urn:ietf:params:scim:api:messages:2.0:ListResponse"
    ],
    "startIndex": 1,
    "totalResults": 1
}
```

### 사용자 생성

- **엔드포인트**: **`<host-url>/scim/Users`**
- **메소드**: POST
- **설명**: 새 사용자 리소스를 생성합니다.
- **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| emails | 다중 값 배열 | 예 (기본 이메일 설정 필수) |
| userName | 문자열 | 예 |
- **요청 예제**:

```bash
POST /scim/Users
```

```json
{
  "schemas": [
    "urn:ietf:params:scim:schemas:core:2.0:User"
  ],
  "emails": [
    {
      "primary": true,
      "value": "admin-user2@test.com"
    }
  ],
  "userName": "dev-user2"
}
```

- **응답 예제**:

```bash
(Status 201)
```

```json
{
    "active": true,
    "displayName": "Dev User 2",
    "emails": {
        "Value": "dev-user2@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "def",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "location": "Users/def"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user2"
}
```

### 사용자 삭제

- **엔드포인트**: **`<host-url>/scim/Users/{id}`**
- **메소드**: DELETE
- **설명**: 사용자의 고유 ID를 제공하여 [SaaS 클라우드](../hosting-options/saas_cloud.md) 조직 또는 [전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [셀프 관리 인스턴스](../hosting-options/self-managed.md) 인스턴스에서 사용자를 완전히 삭제합니다. 필요하면 [Create user](#create-user) API를 사용해 조직이나 인스턴스에 다시 사용자를 추가할 수 있습니다.
- **요청 예제**:

:::note
사용자를 일시적으로 비활성화하려면 `PATCH` 엔드포인트를 사용하는 [Deactivate user](#deactivate-user) API를 참조하세요.
:::

```bash
DELETE /scim/Users/abc
```

- **응답 예제**:

```json
(Status 204)
```

### 사용자 비활성화

- **엔드포인트**: **`<host-url>/scim/Users/{id}`**
- **메소드**: PATCH
- **설명**: 사용자의 고유 ID를 제공하여 [전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [셀프 관리 인스턴스](../hosting-options/self-managed.md)에서 사용자를 일시적으로 비활성화합니다. 필요시 [Reactivate user](#reactivate-user) API를 사용해 사용자를 재활성화하세요.
- **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. `replace`만 허용됩니다. |
| value | 오브젝트 | 사용자를 비활성화할 것임을 나타내는 오브젝트 `{"active": false}`. |

:::note
사용자 비활성화 및 재활성화 작업은 [SaaS 클라우드](../hosting-options/saas_cloud.md)에서 지원되지 않습니다.
:::

- **요청 예제**:

```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "value": {"active": false}
        }
    ]
}
```

- **응답 예제**:
이것은 User 오브젝트를 반환합니다.

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1"
}
```

### 사용자 재활성화

- **엔드포인트**: **`<host-url>/scim/Users/{id}`**
- **메소드**: PATCH
- **설명**: 사용자의 고유 ID를 제공하여 [전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [셀프 관리 인스턴스](../hosting-options/self-managed.md)에서 비활성화된 사용자를 재활성화합니다.
- **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. `replace`만 허용됩니다. |
| value | 오브젝트 | 사용자를 재활성화할 것임을 나타내는 오브젝트 `{"active": true}`. |

:::note
사용자 비활성화 및 재활성화 작업은 [SaaS 클라우드](../hosting-options/saas_cloud.md)에서 지원되지 않습니다.
:::

- **요청 예제**:

```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "value": {"active": true}
        }
    ]
}
```

- **응답 예제**:
이것은 User 오브젝트를 반환합니다.

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1"
}
```

### 조직 레벨 역할 사용자에게 할당

- **엔드포인트**: **`<host-url>/scim/Users/{id}`**
- **메소드**: PATCH
- **설명**: 사용자에게 조직 레벨 역할을 할당합니다. 역할은 [여기](./manage-users#invite-users)에 설명된 것처럼 `admin`, `viewer`, 또는 `member` 중 하나일 수 있습니다. [SaaS 클라우드](../hosting-options/saas_cloud.md)의 경우, SCIM API에 대해 올바른 조직이 사용자 설정에서 구성되었는지 확인하세요.
- **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. `replace`만 허용됩니다. |
| path | 문자열 | 역할 할당 작업이 적용되는 범위. `organizationRole`만 허용됩니다. |
| value | 문자열 | 사용자에게 할당할 사전 정의된 조직 레벨 역할. `admin`, `viewer`, 또는 `member` 중 하나일 수 있습니다. 이 필드는 사전 정의된 역할에 대해 대소문자를 구분하지 않습니다. |
- **요청 예제**:

```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "organizationRole",
            "value": "admin" // 사용자의 조직 범위 역할을 admin으로 설정
        }
    ]
}
```

- **응답 예제**:
이것은 User 오브젝트를 반환합니다.

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1",
    "teamRoles": [  // 사용자가 속한 모든 팀에서의 역할을 반환
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 조직 범위에서의 사용자의 역할을 반환
}
```

### 팀 레벨 역할 사용자에게 할당

- **엔드포인트**: **`<host-url>/scim/Users/{id}`**
- **메소드**: PATCH
- **설명**: 사용자에게 팀 레벨 역할을 할당합니다. 역할은 [여기](./manage-users#team-roles)에 설명된 것처럼 `admin`, `viewer`, `member` 또는 커스텀 역할 중 하나일 수 있습니다. [SaaS 클라우드](../hosting-options/saas_cloud.md)의 경우, SCIM API에 대해 올바른 조직이 사용자 설정에서 구성되었는지 확인하세요.
- **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. `replace`만 허용됩니다. |
| path | 문자열 | 역할 할당 작업이 적용되는 범위. `teamRoles`만 허용됩니다. |
| value | 오브젝트 배열 | 오브젝트 배열로, 오브젝트는 `teamName`과 `roleName` 속성을 포함합니다. `teamName`은 사용자가 해당 역할을 가진 팀의 이름이며, `roleName`은 `admin`, `viewer`, `member` 또는 커스텀 역할 중 하나일 수 있습니다. 이 필드는 사전 정의된 역할에 대해 대소문자를 구분하지 않으며, 커스텀 역할에 대해서는 대소문자를 구분합니다. |
- **요청 예제**:

```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "teamRoles",
            "value": [
                {
                    "roleName": "admin", // 사전 정의된 역할의 경우 대소문자 구분하지 않음, 커스텀 역할의 경우 대소문자 구분
                    "teamName": "team1" // 사용자의 team1 팀에서의 역할을 admin으로 설정
                }
            ]
        }
    ]
}
```

- **응답 예제**:
이것은 User 오브젝트를 반환합니다.

```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "dev-user1@test.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Users/abc"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1",
    "teamRoles": [  // 사용자가 속한 모든 팀에서의 역할을 반환
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 조직 범위에서의 사용자의 역할을 반환
}
```

## Group 리소스

SCIM 그룹 리소스는 W&B 팀에 매핑됩니다. 즉, W&B 배포에서 SCIM 그룹을 생성하면 W&B 팀이 생성됩니다. 다른 그룹 엔드포인트도 마찬가지입니다.

### 팀 가져오기

- **엔드포인트**: **`<host-url>/scim/Groups/{id}`**
- **메소드**: GET
- **설명**: 팀의 고유 ID를 제공하여 팀 정보를 검색합니다.
- **요청 예제**:

```bash
GET /scim/Groups/ghi
```

- **응답 예제**:

```bash
(Status 200)
```

```json
{
    "displayName": "wandb-devs",
    "id": "ghi",
    "members": [
        {
            "Value": "abc",
            "Ref": "",
            "Type": "",
            "Display": "dev-user1"
        }
    ],
    "meta": {
        "resourceType": "Group",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Groups/ghi"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:Group"
    ]
}
```

### 팀 목록 가져오기

- **엔드포인트**: **`<host-url>/scim/Groups`**
- **메소드**: GET
- **설명**: 팀 목록을 검색합니다.
- **요청 예제**:

```bash
GET /scim/Groups
```

- **응답 예제**:

```bash
(Status 200)
```

```json
{
    "Resources": [
        {
            "displayName": "wandb-devs",
            "id": "ghi",
            "members": [
                {
                    "Value": "abc",
                    "Ref": "",
                    "Type": "",
                    "Display": "dev-user1"
                }
            ],
            "meta": {
                "resourceType": "Group",
                "created": "2023-10-01T00:00:00Z",
                "lastModified": "2023-10-01T00:00:00Z",
                "location": "Groups/ghi"
            },
            "schemas": [
                "urn:ietf:params:scim:schemas:core:2.0:Group"
            ]
        }
    ],
    "itemsPerPage": 9999,
    "schemas": [
        "urn:ietf:params:scim:api:messages:2.0:ListResponse"
    ],
    "startIndex": 1,
    "totalResults": 1
}
```

### 팀 생성

- **엔드포인트**: **`<host-url>/scim/Groups`**
- **메소드**: POST
- **설명**: 새로운 팀 리소스를 생성합니다.
- **지원되는 필드**:

| 필드         | 유형                  | 필수                        |
|--------------|-----------------------|-----------------------------|
| displayName  | 문자열                | 예                          |
| members      | 다중 값 배열          | 예 (서브 필드 `value`는 사용자 ID에 매핑) |

- **요청 예제**:

`wandb-support`라는 팀을 `dev-user2`를 멤버로 생성합니다.

```bash
POST /scim/Groups
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
    "displayName": "wandb-support",
    "members": [
        {
            "value": "def"
        }
    ]
}
```

- **응답 예제**:

```bash
(Status 201)
```

```json
{
    "displayName": "wandb-support",
    "id": "jkl",
    "members": [
        {
            "Value": "def",
            "Ref": "",
            "Type": "",
            "Display": "dev-user2"
        }
    ],
    "meta": {
        "resourceType": "Group",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:00:00Z",
        "location": "Groups/jkl"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:Group"
    ]
}
```

### 팀 업데이트

- **엔드포인트**: **`<host-url>/scim/Groups/{id}`**
- **메소드**: PATCH
- **설명**: 기존 팀의 멤버십 목록을 업데이트합니다.
- **지원되는 작업**: `add` 멤버, `remove` 멤버
- **요청 예제**:

`wandb-devs`에 `dev-user2`를 추가합니다.

```bash
PATCH /scim/Groups/ghi
```

```json
{
	"schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
	"Operations": [
		{
			"op": "add",
			"path": "members",
			"value": [
	      {
					"value": "def",
				}
	    ]
		}
	]
}
```

- **응답 예제**:

```bash
(Status 200)
```

```json
{
    "displayName": "wandb-devs",
    "id": "ghi",
    "members": [
        {
            "Value": "abc",
            "Ref": "",
            "Type": "",
            "Display": "dev-user1"
        },
        {
            "Value": "def",
            "Ref": "",
            "Type": "",
            "Display": "dev-user2"
        }
    ],
    "meta": {
        "resourceType": "Group",
        "created": "2023-10-01T00:00:00Z",
        "lastModified": "2023-10-01T00:01:00Z",
        "location": "Groups/ghi"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:Group"
    ]
}
```

### 팀 삭제

- SCIM API는 현재 팀 삭제를 지원하지 않습니다. 이는 팀과 연결된 추가 데이터가 있기 때문입니다. 모든 것을 삭제하려는 경우에는 앱에서 팀을 삭제하십시오.

## Role 리소스

SCIM 역할 리소스는 W&B 커스텀 역할에 매핑됩니다. 앞서 언급했듯이, `/Roles` 엔드포인트는 공식 SCIM 스키마의 일부가 아니며, W&B는 W&B 조직에서 커스텀 역할의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가했습니다.

### 커스텀 역할 가져오기

- **엔드포인트:** **`<host-url>/scim/Roles/{id}`**
- **메소드**: GET
- **설명**: 역할의 고유 ID를 제공하여 커스텀 역할 정보를 검색합니다.
- **요청 예제**:

```bash
GET /scim/Roles/abc
```

- **응답 예제**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 사전 정의된 역할을 나타냄
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "Sample custom role",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // 회원 역을 상속함
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 관리자에 의해 추가된 커스텀 권한
        }
    ],
    "schemas": [
        ""
    ]
}
```

### 커스텀 역할 목록

- **엔드포인트:** **`<host-url>/scim/Roles`**
- **메소드**: GET
- **설명**: W&B 조직의 모든 커스텀 역할 정보를 검색합니다.
- **요청 예제**:

```bash
GET /scim/Roles
```

- **응답 예제**:

```bash
(Status 200)
```

```json
{
   "Resources": [
        {
            "description": "A sample custom role for example",
            "id": "Um9sZTo3",
            "inheritedFrom": "member", // 커스텀 역할이 상속하는 사전 정의된 역할을 나타냄
            "meta": {
                "resourceType": "Role",
                "created": "2023-11-20T23:10:14Z",
                "lastModified": "2023-11-20T23:31:23Z",
                "location": "Roles/Um9sZTo3"
            },
            "name": "Sample custom role",
            "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
            "permissions": [
                {
                    "name": "artifact:read",
                    "isInherited": true // 회원 역할로부터 상속됨
                },
                ...
                ...
                {
                    "name": "project:update",
                    "isInherited": false // 관리자에 의해 추가된 커스텀 권한
                }
            ],
            "schemas": [
                ""
            ]
        },
        {
            "description": "Another sample custom role for example",
            "id": "Um9sZToxMg==",
            "inheritedFrom": "viewer", // 커스텀 역할이 상속하는 사전 정의된 역할을 나타냄
            "meta": {
                "resourceType": "Role",
                "created": "2023-11-21T01:07:50Z",
                "location": "Roles/Um9sZToxMg=="
            },
            "name": "Sample custom role 2",
            "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
            "permissions": [
                {
                    "name": "launchagent:read",
                    "isInherited": true // 시청자 역할로부터 상속됨
                },
                ...
                ...
                {
                    "name": "run:stop",
                    "isInherited": false // 관리자에 의해 추가된 커스텀 권한
                }
            ],
            "schemas": [
                ""
            ]
        }
    ],
    "itemsPerPage": 9999,
    "schemas": [
        "urn:ietf:params:scim:api:messages:2.0:ListResponse"
    ],
    "startIndex": 1,
    "totalResults": 2
}
```

### 커스텀 역할 생성

- **엔드포인트**: **`<host-url>/scim/Roles`**
- **메소드**: POST
- **설명**: W&B 조직에 새로운 커스텀 역할을 생성합니다.
- **지원되는 필드**:

| 필드            | 유형          | 필수        |
|-----------------|---------------|-------------|
| name            | 문자열        | 커스텀 역할의 이름 |
| description     | 문자열        | 커스텀 역할의 설명 |
| permissions     | 오브젝트 배열 | `w&bobject:operation` 형식의 값을 가진 `name` 문자열 필드가 있는 권한 오브젝트 배열 예: W&B 실행에서 삭제 작업의 경우 권한 오브젝트의 `name`은 `run:delete`가 됩니다. |
| inheritedFrom   | 문자열        | 커스텀 역할이 상속할 사전 정의된 역할. `member` 또는 `viewer`일 수 있습니다. |

- **요청 예제**:

```bash
POST /scim/Roles
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
    "name": "Sample custom role",
    "description": "A sample custom role for example",
    "permissions": [
        {
            "name": "project:update"
        }
    ],
    "inheritedFrom": "member"
}
```

- **응답 예제**:

```bash
(Status 201)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 사전 정의된 역할을 나타냄
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "Sample custom role",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // 회원 역할로부터 상속됨
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 관리자에 의해 추가된 커스텀 권한
        }
    ],
    "schemas": [
        ""
    ]
}
```

### 커스텀 역할 삭제

- **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
- **메소드**: DELETE
- **설명**: W&B 조직에서 커스텀 역할을 삭제합니다. **신중히 사용하세요.** 커스텀 역할 삭제 전 해당 역할에 할당된 모든 사용자에게 상속된 사전 정의된 역할이 이제 할당됩니다.
- **요청 예제**:

```bash
DELETE /scim/Roles/abc
```

- **응답 예제**:

```bash
(Status 204)
```

### 커스텀 역할 권한 업데이트

- **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
- **메소드**: PATCH
- **설명**: W&B 조직의 커스텀 역할에서 커스텀 권한을 추가 또는 제거합니다.
- **지원되는 필드**:

| 필드         | 유형        | 필수          |
|--------------|-------------|---------------|
| operations   | 오브젝트 배열| 작업 오브젝트 배열 |
| op           | 문자열      | 작업 오브젝트 내의 작업 유형 `add` 또는 `remove`일 수 있음 |
| path         | 문자열      | 작업 오브젝트 내의 정적 필드, 유일한 허용 값은 `permissions` |
| value        | 오브젝트 배열| 각 오브젝트는 `name` 문자열 필드를 포함하는 권한 오브젝트 배열 W&B 실행에서 삭제 작업의 권한 오브젝트는 `name`으로 `run:delete`를 가질 것입니다. |

- **요청 예제**:

```bash
PATCH /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add", // 작업 유형을 나타내며 다른 가능한 값은 `remove`입니다
            "path": "permissions",
            "value": [
                {
                    "name": "project:delete"
                }
            ]
        }
    ]
}
```

- **응답 예제**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 사전 정의된 역할을 나타냄
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "Sample custom role",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // 회원 역할로부터 상속됨
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 업데이트 전 관리자에 의해 추가된 기존 커스텀 권한
        },
        {
            "name": "project:delete",
            "isInherited": false // 업데이트의 일부로 관리자에 의해 추가된 새 커스텀 권한
        }
    ],
    "schemas": [
        ""
    ]
}
```

### 커스텀 역할 메타데이터 업데이트

- **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
- **메소드**: PUT
- **설명**: W&B 조직의 커스텀 역할의 이름, 설명 또는 상속된 역할을 업데이트합니다. 이 작업은 커스텀 역할 내의 기존, 즉, 상속되지 않은 커스텀 권한에는 영향을 미치지 않습니다.
- **지원되는 필드**:

| 필드          | 유형   | 필수       |
|---------------|--------|------------|
| name          | 문자열 | 커스텀 역할의 이름 |
| description   | 문자열 | 커스텀 역할의 설명 |
| inheritedFrom | 문자열 | 커스텀 역할이 상속하는 사전 정의된 역할. `member` 또는 `viewer`일 수 있습니다. |

- **요청 예제**:

```bash
PUT /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
    "name": "Sample custom role",
    "description": "A sample custom role for example but now based on viewer",
    "inheritedFrom": "viewer"
}
```

- **응답 예제**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example but now based on viewer", // 요청에 따라 변경된 설명
    "id": "Um9sZTo3",
    "inheritedFrom": "viewer", // 요청에 따라 변경된 사전 정의된 역할을 나타냄
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "Sample custom role",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // 시청자 역할로부터 상속됨
        },
        ... // 업데이트 후 시청자 역할에 없는 멤버 역할의 모든 권한은 더 이상 상속되지 않음
        {
            "name": "project:update",
            "isInherited": false // 관리자에 의해 추가된 커스텀 권한
        },
        {
            "name": "project:delete",
            "isInherited": false // 관리자에 의해 추가된 커스텀 권한
        }
    ],
    "schemas": [
        ""
    ]
}
```