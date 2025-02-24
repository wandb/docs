---
title: Manage users, groups, and roles with SCIM
menu:
  default:
    identifier: ko-guides-hosting-iam-scim
    parent: identity-and-access-management-iam
weight: 4
---

System for Cross-domain Identity Management (SCIM) API를 사용하면 인스턴스 또는 organization 관리자가 W&B organization에서 user, group, custom role을 관리할 수 있습니다. SCIM group은 W&B Teams에 매핑됩니다.

SCIM API는 `<host-url>/scim/`에서 엑세스할 수 있으며 [RC7643 프로토콜](https://www.rfc-editor.org/rfc/rfc7643)에서 찾을 수 있는 필드의 서브셋을 사용하여 `/Users` 및 `/Groups` 엔드포인트를 지원합니다. 또한 공식 SCIM 스키마의 일부가 아닌 `/Roles` 엔드포인트도 포함합니다. W&B는 W&B organization에서 custom role의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가합니다.

{{% alert %}}
SCIM API는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}), [자체 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}), [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})를 포함한 모든 호스팅 옵션에 적용됩니다. SaaS 클라우드에서 organization 관리자는 SCIM API 요청이 올바른 organization으로 전달되도록 user 설정에서 기본 organization을 구성해야 합니다. 이 설정은 user 설정 내의 `SCIM API Organization` 섹션에서 사용할 수 있습니다.
{{% /alert %}}

## 인증

organization 또는 인스턴스 관리자는 API 키로 기본 인증을 사용하여 SCIM API에 엑세스할 수 있습니다. HTTP 요청의 `Authorization` 헤더를 `Basic` 문자열 다음에 공백, 그리고 `username:API-KEY` 형식의 base-64로 인코딩된 문자열로 설정합니다. 즉, 사용자 이름과 API 키를 `:` 문자로 구분된 값으로 바꾸고 결과를 base-64로 인코딩합니다. 예를 들어 `demo:p@55w0rd`로 인증하려면 헤더는 `Authorization: Basic ZGVtbzpwQDU1dzByZA==`여야 합니다.

## User 리소스

SCIM user 리소스는 W&B Users에 매핑됩니다.

### User 가져오기

-   **엔드포인트:** **`<host-url>/scim/Users/{id}`**
-   **메서드**: GET
-   **설명**: user의 고유 ID를 제공하여 [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) organization 또는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 특정 user에 대한 정보를 검색합니다.
-   **요청 예시**:

```bash
GET /scim/Users/abc
```

-   **응답 예시**:

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

### User 나열하기

-   **엔드포인트:** **`<host-url>/scim/Users`**
-   **메서드**: GET
-   **설명**: [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) organization 또는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에 있는 모든 Users 목록을 검색합니다.
-   **요청 예시**:

```bash
GET /scim/Users
```

-   **응답 예시**:

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

### User 만들기

-   **엔드포인트**: **`<host-url>/scim/Users`**
-   **메서드**: POST
-   **설명**: 새 user 리소스를 만듭니다.
-   **지원되는 필드**:

| 필드        | 유형               | 필수 |
| ----------- | ------------------ | ---- |
| emails      | 다중 값 어레이       | 예 (`primary` 이메일이 설정되었는지 확인) |
| userName    | 문자열               | 예   |

-   **요청 예시**:

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

-   **응답 예시**:

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

### User 삭제

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: DELETE
-   **설명**: user의 고유 ID를 제공하여 [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) organization 또는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 user를 완전히 삭제합니다. 필요한 경우 [User 만들기]({{< relref path="#create-user" lang="ko" >}}) API를 사용하여 organization 또는 인스턴스에 user를 다시 추가합니다.
-   **요청 예시**:

{{% alert %}}
user를 일시적으로 비활성화하려면 `PATCH` 엔드포인트를 사용하는 [User 비활성화]({{< relref path="#deactivate-user" lang="ko" >}}) API를 참조하십시오.
{{% /alert %}}

```bash
DELETE /scim/Users/abc
```

-   **응답 예시**:

```json
(Status 204)
```

### User 비활성화

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: PATCH
-   **설명**: user의 고유 ID를 제공하여 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 user를 일시적으로 비활성화합니다. 필요한 경우 [User 다시 활성화]({{< relref path="#reactivate-user" lang="ko" >}}) API를 사용하여 user를 다시 활성화합니다.
-   **지원되는 필드**:

| 필드    | 유형     | 필수 |
| ------- | -------- | ---- |
| op      | 문자열   | 작업 유형. 허용되는 유일한 값은 `replace`입니다. |
| value   | 오브젝트 | user를 비활성화해야 함을 나타내는 오브젝트 `{"active": false}`입니다. |

{{% alert %}}
User 비활성화 및 재활성화 작업은 [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서 지원되지 않습니다.
{{% /alert %}}

-   **요청 예시**:

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

-   **응답 예시**:
User 오브젝트를 반환합니다.

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

### User 다시 활성화

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: PATCH
-   **설명**: user의 고유 ID를 제공하여 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 비활성화된 user를 다시 활성화합니다.
-   **지원되는 필드**:

| 필드    | 유형     | 필수 |
| ------- | -------- | ---- |
| op      | 문자열   | 작업 유형. 허용되는 유일한 값은 `replace`입니다. |
| value   | 오브젝트 | user를 다시 활성화해야 함을 나타내는 오브젝트 `{"active": true}`입니다. |

{{% alert %}}
User 비활성화 및 재활성화 작업은 [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서 지원되지 않습니다.
{{% /alert %}}

-   **요청 예시**:

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

-   **응답 예시**:
User 오브젝트를 반환합니다.

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

### User에게 organization 수준 Role 할당

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: PATCH
-   **설명**: user에게 organization 수준 Role을 할당합니다. Role은 [여기]({{< relref path="access-management/manage-organization.md#invite-a-user" lang="ko" >}})에 설명된 대로 `admin`, `viewer` 또는 `member` 중 하나일 수 있습니다. [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우 user 설정에서 SCIM API에 대한 올바른 organization을 구성했는지 확인하십시오.
-   **지원되는 필드**:

| 필드               | 유형     | 필수 |
| ------------------ | -------- | ---- |
| op                 | 문자열   | 작업 유형. 허용되는 유일한 값은 `replace`입니다. |
| path               | 문자열   | Role 할당 작업이 적용되는 범위입니다. 허용되는 유일한 값은 `organizationRole`입니다. |
| value              | 문자열   | user에게 할당할 미리 정의된 organization 수준 Role입니다. `admin`, `viewer` 또는 `member` 중 하나일 수 있습니다. 이 필드는 미리 정의된 Role에 대해 대소문자를 구분하지 않습니다. |

-   **요청 예시**:

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
            "value": "admin" // will set the user's organization-scoped role to admin
        }
    ]
}
```

-   **응답 예시**:
User 오브젝트를 반환합니다.

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
    "teamRoles": [  // Returns the user's roles in all the teams that they are a part of
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // Returns the user's role at the organization scope
}
```

### User에게 팀 수준 Role 할당

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: PATCH
-   **설명**: user에게 팀 수준 Role을 할당합니다. Role은 [여기]({{< relref path="access-management/manage-organization.md#assign-or-update-a-team-members-role" lang="ko" >}})에 설명된 대로 `admin`, `viewer`, `member` 또는 custom role 중 하나일 수 있습니다. [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우 user 설정에서 SCIM API에 대한 올바른 organization을 구성했는지 확인하십시오.
-   **지원되는 필드**:

| 필드    | 유형           | 필수 |
| ------- | -------------- | ---- |
| op      | 문자열         | 작업 유형. 허용되는 유일한 값은 `replace`입니다. |
| path    | 문자열         | Role 할당 작업이 적용되는 범위입니다. 허용되는 유일한 값은 `teamRoles`입니다. |
| value   | 오브젝트 어레이 | 오브젝트가 `teamName` 및 `roleName` 속성으로 구성된 단일 오브젝트 어레이입니다. `teamName`은 user가 Role을 보유하는 팀의 이름이고 `roleName`은 `admin`, `viewer`, `member` 또는 custom role 중 하나일 수 있습니다. 이 필드는 미리 정의된 Role에 대해 대소문자를 구분하지 않고 custom role에 대해 대소문자를 구분합니다. |

-   **요청 예시**:

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
                    "roleName": "admin", // role name is case insensitive for predefined roles and case sensitive for custom roles
                    "teamName": "team1" // will set the user's role in the team team1 to admin
                }
            ]
        }
    ]
}
```

-   **응답 예시**:
User 오브젝트를 반환합니다.

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
    "teamRoles": [  // Returns the user's roles in all the teams that they are a part of
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // Returns the user's role at the organization scope
}
```

## Group 리소스

SCIM group 리소스는 W&B Teams에 매핑됩니다. 즉, W&B 배포에서 SCIM group을 만들면 W&B Team이 생성됩니다. 다른 group 엔드포인트에도 동일하게 적용됩니다.

### Team 가져오기

-   **엔드포인트**: **`<host-url>/scim/Groups/{id}`**
-   **메서드**: GET
-   **설명**: team의 고유 ID를 제공하여 team 정보를 검색합니다.
-   **요청 예시**:

```bash
GET /scim/Groups/ghi
```

-   **응답 예시**:

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

### Team 나열하기

-   **엔드포인트**: **`<host-url>/scim/Groups`**
-   **메서드**: GET
-   **설명**: team 목록을 검색합니다.
-   **요청 예시**:

```bash
GET /scim/Groups
```

-   **응답 예시**:

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

### Team 만들기

-   **엔드포인트**: **`<host-url>/scim/Groups`**
-   **메서드**: POST
-   **설명**: 새 team 리소스를 만듭니다.
-   **지원되는 필드**:

| 필드        | 유형               | 필수 |
| ----------- | ------------------ | ---- |
| displayName | 문자열               | 예   |
| members     | 다중 값 어레이       | 예 (하위 필드 `value`는 필수이며 user ID에 매핑됩니다.) |

-   **요청 예시**:

`dev-user2`를 멤버로 포함하는 `wandb-support`라는 team 만들기

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

-   **응답 예시**:

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

### Team 업데이트

-   **엔드포인트**: **`<host-url>/scim/Groups/{id}`**
-   **메서드**: PATCH
-   **설명**: 기존 team의 멤버십 목록을 업데이트합니다.
-   **지원되는 작업**: 멤버 `add`, 멤버 `remove`
-   **요청 예시**:

`wandb-devs`에 `dev-user2` 추가

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

-   **응답 예시**:

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

### Team 삭제

-   Team 삭제는 team에 연결된 추가 data가 있으므로 현재 SCIM API에서 지원되지 않습니다. 모든 항목을 삭제할지 확인하려면 앱에서 team을 삭제하십시오.

## Role 리소스

SCIM role 리소스는 W&B custom role에 매핑됩니다. 앞에서 언급했듯이 `/Roles` 엔드포인트는 공식 SCIM 스키마의 일부가 아니며 W&B는 W&B organization에서 custom role의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가합니다.

### Custom role 가져오기

-   **엔드포인트:** **`<host-url>/scim/Roles/{id}`**
-   **메서드**: GET
-   **설명**: role의 고유 ID를 제공하여 custom role에 대한 정보를 검색합니다.
-   **요청 예시**:

```bash
GET /scim/Roles/abc
```

-   **응답 예시**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // indicates the predefined role
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
            "isInherited": true // inherited from member predefined role
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // custom permission added by admin
        }
    ],
    "schemas": [
        ""
    ]
}
```

### Custom role 나열하기

-   **엔드포인트:** **`<host-url>/scim/Roles`**
-   **메서드**: GET
-   **설명**: W&B organization의 모든 custom role에 대한 정보를 검색합니다.
-   **요청 예시**:

```bash
GET /scim/Roles
```

-   **응답 예시**:

```bash
(Status 200)
```

```json
{
   "Resources": [
        {
            "description": "A sample custom role for example",
            "id": "Um9sZTo3",
            "inheritedFrom": "member", // indicates the predefined role that the custom role inherits from
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
                    "isInherited": true // inherited from member predefined role
                },
                ...
                ...
                {
                    "name": "project:update",
                    "isInherited": false // custom permission added by admin
                }
            ],
            "schemas": [
                ""
            ]
        },
        {
            "description": "Another sample custom role for example",
            "id": "Um9sZToxMg==",
            "inheritedFrom": "viewer", // indicates the predefined role that the custom role inherits from
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
                    "isInherited": true // inherited from viewer predefined role
                },
                ...
                ...
                {
                    "name": "run:stop",
                    "isInherited": false // custom permission added by admin
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

### Custom role 만들기

-   **엔드포인트**: **`<host-url>/scim/Roles`**
-   **메서드**: POST
-   **설명**: W&B organization에 새 custom role을 만듭니다.
-   **지원되는 필드**:

| 필드          | 유형           | 필수 |
| ------------- | -------------- | ---- |
| name          | 문자열         | custom role의 이름 |
| description   | 문자열         | custom role에 대한 설명 |
| permissions   | 오브젝트 어레이 | 각 오브젝트에 `w&bobject:operation` 형식의 값을 갖는 `name` 문자열 필드가 포함된 권한 오브젝트의 어레이입니다. 예를 들어 W&B Runs에 대한 삭제 작업을 위한 권한 오브젝트의 `name`은 `run:delete`가 됩니다. |
| inheritedFrom | 문자열         | custom role이 상속할 미리 정의된 Role입니다. `member` 또는 `viewer`일 수 있습니다. |

-   **요청 예시**:

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

-   **응답 예시**:

```bash
(Status 201)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // indicates the predefined role
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
            "isInherited": true // inherited from member predefined role
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // custom permission added by admin
        }
    ],
    "schemas": [
        ""
    ]
}
```

### Custom role 삭제

-   **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
-   **메서드**: DELETE
-   **설명**: W&B organization에서 custom role을 삭제합니다. **주의해서 사용하십시오**. custom role이 상속된 미리 정의된 Role이 이제 작업 전에 custom role이 할당된 모든 Users에게 할당됩니다.
-   **요청 예시**:

```bash
DELETE /scim/Roles/abc
```

-   **응답 예시**:

```bash
(Status 204)
```

### Custom role 권한 업데이트

-   **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
-   **메서드**: PATCH
-   **설명**: W&B organization의 custom role에서 custom 권한을 추가하거나 제거합니다.
-   **지원되는 필드**:

| 필드        | 유형           | 필수 |
| ----------- | -------------- | ---- |
| operations  | 오브젝트 어레이 | 작업 오브젝트의 어레이 |
| op          | 문자열         | 작업 오브젝트 내의 작업 유형입니다. `add` 또는 `remove`일 수 있습니다. |
| path        | 문자열         | 작업 오브젝트의 정적 필드입니다. 허용되는 유일한 값은 `permissions`입니다. |
| value       | 오브젝트 어레이 | 각 오브젝트에 `w&bobject:operation` 형식의 값을 갖는 `name` 문자열 필드가 포함된 권한 오브젝트의 어레이입니다. 예를 들어 W&B Runs에 대한 삭제 작업을 위한 권한 오브젝트의 `name`은 `run:delete`가 됩니다. |

-   **요청 예시**:

```bash
PATCH /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add", // indicates the type of operation, other possible value being `remove`
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

-   **응답 예시**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // indicates the predefined role
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
            "isInherited": true // inherited from member predefined role
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // existing custom permission added by admin before the update
        },
        {
            "name": "project:delete",
            "isInherited": false // new custom permission added by admin as part of the update
        }
    ],
    "schemas": [
        ""
    ]
}
```

### Custom role 메타데이터 업데이트

-   **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
-   **메서드**: PUT
-   **설명**: W&B organization에서 custom role의 이름, 설명 또는 상속된 Role을 업데이트합니다. 이 작업은 custom role의 기존 custom 권한(즉, 상속되지 않은 custom 권한)에 영향을 주지 않습니다.
-   **지원되는 필드**:

| 필드          | 유형           | 필수 |
| ------------- | -------------- | ---- |
| name          | 문자열         | custom role의 이름 |
| description   | 문자열         | custom role에 대한 설명 |
| inheritedFrom | 문자열         | custom role이 상속할 미리 정의된 Role입니다. `member` 또는 `viewer`일 수 있습니다. |

-   **요청 예시**:

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

-   **응답 예시**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example but now based on viewer", // changed the descripton per the request
    "id": "Um9sZTo3",
    "inheritedFrom": "viewer", // indicates the predefined role which is changed per the request
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
            "isInherited": true // inherited from viewer predefined role
        },
        ... // Any permissions that are in member predefined role but not in viewer will not be inherited post the update
        {
            "name": "project:update",
            "isInherited": false // custom permission added by admin
        },
        {
            "name": "project:delete",
            "isInherited": false // custom permission added by admin
        }
    ],
    "schemas": [
        ""
    ]
}
```