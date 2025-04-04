---
title: Manage users, groups, and roles with SCIM
menu:
  default:
    identifier: ko-guides-hosting-iam-scim
    parent: identity-and-access-management-iam
weight: 4
---

{{% alert %}}
[SCIM이 작동하는 것을 보여주는 비디오](https://www.youtube.com/watch?v=Nw3QBqV0I-o) (12분)를 시청하세요.
{{% /alert %}}

SCIM(System for Cross-domain Identity Management) API를 통해 인스턴스 또는 organization 관리자는 W&B organization에서 사용자, 그룹 및 사용자 지정 역할을 관리할 수 있습니다. SCIM 그룹은 W&B Teams에 매핑됩니다.

SCIM API는 `<host-url>/scim/`에서 액세스할 수 있으며 [RC7643 프로토콜](https://www.rfc-editor.org/rfc/rfc7643)에서 찾을 수 있는 필드의 서브셋으로 `/Users` 및 `/Groups` 엔드포인트를 지원합니다. 또한 공식 SCIM 스키마의 일부가 아닌 `/Roles` 엔드포인트도 포함합니다. W&B는 W&B organization에서 사용자 지정 역할의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가합니다.

{{% alert %}}
여러 Enterprise [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) organization의 관리자인 경우 SCIM API 요청이 전송되는 organization을 구성해야 합니다. 프로필 이미지를 클릭한 다음 **User Settings**를 클릭합니다. 설정 이름은 **Default API organization**입니다. 이는 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}), [Self-managed instances]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 및 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})를 포함한 모든 호스팅 옵션에 필요합니다. SaaS Cloud에서 organization 관리자는 SCIM API 요청이 올바른 organization으로 이동하도록 사용자 설정에서 기본 organization을 구성해야 합니다.

선택한 호스팅 옵션은 이 페이지의 예제에서 사용되는 `<host-url>` 자리 표시자의 값을 결정합니다.

또한 예제에서는 `abc` 및 `def`와 같은 사용자 ID를 사용합니다. 실제 요청 및 응답에는 사용자 ID에 대한 해시된 값이 있습니다.
{{% /alert %}}

## 인증

Organization 또는 인스턴스 관리자는 API 키로 기본 인증을 사용하여 SCIM API에 액세스할 수 있습니다. HTTP 요청의 `Authorization` 헤더를 `Basic` 문자열 뒤에 공백, 그런 다음 `username:API-KEY` 형식으로 base-64로 인코딩된 문자열로 설정합니다. 즉, 사용자 이름과 API 키를 `:` 문자로 구분된 값으로 바꾸고 결과를 base-64로 인코딩합니다. 예를 들어 `demo:p@55w0rd`로 인증하려면 헤더는 `Authorization: Basic ZGVtbzpwQDU1dzByZA==`여야 합니다.

## 사용자 리소스

SCIM 사용자 리소스는 W&B Users에 매핑됩니다.

### 사용자 가져오기

-   **엔드포인트:** **`<host-url>/scim/Users/{id}`**
-   **메서드**: GET
-   **설명**: 사용자 고유 ID를 제공하여 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) organization 또는 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 특정 사용자에 대한 정보를 검색합니다.
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

### 사용자 목록

-   **엔드포인트:** **`<host-url>/scim/Users`**
-   **메서드**: GET
-   **설명**: [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) organization 또는 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 모든 사용자 목록을 검색합니다.
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

### 사용자 생성

-   **엔드포인트**: **`<host-url>/scim/Users`**
-   **메서드**: POST
-   **설명**: 새 사용자 리소스를 만듭니다.
-   **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| emails | 다중 값 배열 | 예( `primary` 이메일이 설정되었는지 확인) |
| userName | 문자열 | 예 |

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

### 사용자 삭제

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: DELETE
-   **설명**: 사용자 고유 ID를 제공하여 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) organization 또는 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 사용자를 완전히 삭제합니다. 필요한 경우 [사용자 생성]({{< relref path="#create-user" lang="ko" >}}) API를 사용하여 사용자를 organization 또는 인스턴스에 다시 추가합니다.
-   **요청 예시**:

{{% alert %}}
사용자를 일시적으로 비활성화하려면 `PATCH` 엔드포인트를 사용하는 [사용자 비활성화]({{< relref path="#deactivate-user" lang="ko" >}}) API를 참조하세요.
{{% /alert %}}

```bash
DELETE /scim/Users/abc
```

-   **응답 예시**:

```json
(Status 204)
```

### 사용자 비활성화

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: PATCH
-   **설명**: 사용자 고유 ID를 제공하여 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 사용자를 일시적으로 비활성화합니다. 필요한 경우 [사용자 다시 활성화]({{< relref path="#reactivate-user" lang="ko" >}}) API를 사용하여 사용자를 다시 활성화합니다.
-   **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. 허용되는 유일한 값은 `replace`입니다. |
| value | 오브젝트 | 사용자를 비활성화해야 함을 나타내는 오브젝트 `{"active": false}`입니다. |

{{% alert %}}
사용자 비활성화 및 재활성화 작업은 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서 지원되지 않습니다.
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

### 사용자 다시 활성화

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: PATCH
-   **설명**: 사용자 고유 ID를 제공하여 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서 비활성화된 사용자를 다시 활성화합니다.
-   **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. 허용되는 유일한 값은 `replace`입니다. |
| value | 오브젝트 | 사용자를 다시 활성화해야 함을 나타내는 오브젝트 `{"active": true}`입니다. |

{{% alert %}}
사용자 비활성화 및 재활성화 작업은 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서 지원되지 않습니다.
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

### 사용자에게 organization 수준 역할 할당

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: PATCH
-   **설명**: 사용자에게 organization 수준 역할을 할당합니다. 역할은 [여기]({{< relref path="access-management/manage-organization.md#invite-a-user" lang="ko" >}})에 설명된 대로 `admin`, `viewer` 또는 `member` 중 하나일 수 있습니다. [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우 사용자 설정에서 SCIM API에 대한 올바른 organization을 구성했는지 확인합니다.
-   **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. 허용되는 유일한 값은 `replace`입니다. |
| path | 문자열 | 역할 할당 작업이 적용되는 범위입니다. 허용되는 유일한 값은 `organizationRole`입니다. |
| value | 문자열 | 사용자에게 할당할 미리 정의된 organization 수준 역할입니다. `admin`, `viewer` 또는 `member` 중 하나일 수 있습니다. 이 필드는 미리 정의된 역할에 대해 대소문자를 구분하지 않습니다. |

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
            "value": "admin" // 사용자의 organization 범위 역할을 admin으로 설정합니다.
        }
    ]
}
```

-   **응답 예시**:
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
    "teamRoles": [  // 사용자가 속한 모든 Teams에서 사용자의 역할을 반환합니다.
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // organization 범위에서 사용자의 역할을 반환합니다.
}
```

### 사용자에게 팀 수준 역할 할당

-   **엔드포인트**: **`<host-url>/scim/Users/{id}`**
-   **메서드**: PATCH
-   **설명**: 사용자에게 팀 수준 역할을 할당합니다. 역할은 [여기]({{< relref path="access-management/manage-organization.md#assign-or-update-a-team-members-role" lang="ko" >}})에 설명된 대로 `admin`, `viewer`, `member` 또는 사용자 지정 역할 중 하나일 수 있습니다. [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우 사용자 설정에서 SCIM API에 대한 올바른 organization을 구성했는지 확인합니다.
-   **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. 허용되는 유일한 값은 `replace`입니다. |
| path | 문자열 | 역할 할당 작업이 적용되는 범위입니다. 허용되는 유일한 값은 `teamRoles`입니다. |
| value | 오브젝트 배열 | 오브젝트가 `teamName` 및 `roleName` 속성으로 구성된 단일 오브젝트 배열입니다. `teamName`은 사용자가 역할을 보유하는 팀의 이름이고, `roleName`은 `admin`, `viewer`, `member` 또는 사용자 지정 역할 중 하나일 수 있습니다. 이 필드는 미리 정의된 역할에 대해 대소문자를 구분하지 않고 사용자 지정 역할에 대해 대소문자를 구분합니다. |

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
                    "roleName": "admin", // 역할 이름은 미리 정의된 역할에 대해 대소문자를 구분하지 않고 사용자 지정 역할에 대해 대소문자를 구분합니다.
                    "teamName": "team1" // 팀 team1에서 사용자의 역할을 admin으로 설정합니다.
                }
            ]
        }
    ]
}
```

-   **응답 예시**:
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
    "teamRoles": [  // 사용자가 속한 모든 Teams에서 사용자의 역할을 반환합니다.
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // organization 범위에서 사용자의 역할을 반환합니다.
}
```

## 그룹 리소스

SCIM 그룹 리소스는 W&B Teams에 매핑됩니다. 즉, W&B 배포에서 SCIM 그룹을 만들면 W&B Team이 생성됩니다. 다른 그룹 엔드포인트에도 동일하게 적용됩니다.

### 팀 가져오기

-   **엔드포인트**: **`<host-url>/scim/Groups/{id}`**
-   **메서드**: GET
-   **설명**: 팀의 고유 ID를 제공하여 팀 정보를 검색합니다.
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

### 팀 목록

-   **엔드포인트**: **`<host-url>/scim/Groups`**
-   **메서드**: GET
-   **설명**: 팀 목록을 검색합니다.
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

### 팀 생성

-   **엔드포인트**: **`<host-url>/scim/Groups`**
-   **메서드**: POST
-   **설명**: 새 팀 리소스를 만듭니다.
-   **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| displayName | 문자열 | 예 |
| members | 다중 값 배열 | 예( `value` 하위 필드는 필수이며 사용자 ID에 매핑됨) |

-   **요청 예시**:

`dev-user2`를 멤버로 하여 `wandb-support`라는 팀을 만듭니다.

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

### 팀 업데이트

-   **엔드포인트**: **`<host-url>/scim/Groups/{id}`**
-   **메서드**: PATCH
-   **설명**: 기존 팀의 멤버십 목록을 업데이트합니다.
-   **지원되는 작업**: 멤버 `add`, 멤버 `remove`
-   **요청 예시**:

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

### 팀 삭제

-   팀 삭제는 현재 SCIM API에서 지원되지 않습니다. 팀에 연결된 추가 데이터가 있기 때문입니다. 모든 항목을 삭제할지 확인하려면 앱에서 팀을 삭제하세요.

## 역할 리소스

SCIM 역할 리소스는 W&B 사용자 지정 역할에 매핑됩니다. 앞에서 언급했듯이 `/Roles` 엔드포인트는 공식 SCIM 스키마의 일부가 아니며 W&B는 W&B organization에서 사용자 지정 역할의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가합니다.

### 사용자 지정 역할 가져오기

-   **엔드포인트:** **`<host-url>/scim/Roles/{id}`**
-   **메서드**: GET
-   **설명**: 역할의 고유 ID를 제공하여 사용자 지정 역할에 대한 정보를 검색합니다.
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

### 사용자 지정 역할 목록

-   **엔드포인트:** **`<host-url>/scim/Roles`**
-   **메서드**: GET
-   **설명**: W&B organization의 모든 사용자 지정 역할에 대한 정보를 검색합니다.
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

### 사용자 지정 역할 생성

-   **엔드포인트**: **`<host-url>/scim/Roles`**
-   **메서드**: POST
-   **설명**: W&B organization에서 새 사용자 지정 역할을 만듭니다.
-   **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| name | 문자열 | 사용자 지정 역할의 이름 |
| description | 문자열 | 사용자 지정 역할에 대한 설명 |
| permissions | 오브젝트 배열 | 각 오브젝트에 `w&bobject:operation` 형식의 값이 있는 `name` 문자열 필드가 포함된 권한 오브젝트 배열입니다. 예를 들어 W&B Runs에 대한 삭제 작업에 대한 권한 오브젝트의 `name`은 `run:delete`입니다. |
| inheritedFrom | 문자열 | 사용자 지정 역할이 상속할 미리 정의된 역할입니다. `member` 또는 `viewer`일 수 있습니다. |

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

### 사용자 지정 역할 삭제

-   **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
-   **메서드**: DELETE
-   **설명**: W&B organization에서 사용자 지정 역할을 삭제합니다. **주의해서 사용하세요**. 사용자 지정 역할이 상속된 미리 정의된 역할은 이제 작업 전에 사용자 지정 역할이 할당된 모든 사용자에게 할당됩니다.
-   **요청 예시**:

```bash
DELETE /scim/Roles/abc
```

-   **응답 예시**:

```bash
(Status 204)
```

### 사용자 지정 역할 권한 업데이트

-   **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
-   **메서드**: PATCH
-   **설명**: W&B organization에서 사용자 지정 역할에 사용자 지정 권한을 추가하거나 제거합니다.
-   **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| operations | 오브젝트 배열 | 작업 오브젝트 배열 |
| op | 문자열 | 작업 오브젝트 내의 작업 유형입니다. `add` 또는 `remove`일 수 있습니다. |
| path | 문자열 | 작업 오브젝트의 정적 필드입니다. 허용되는 유일한 값은 `permissions`입니다. |
| value | 오브젝트 배열 | 각 오브젝트에 `w&bobject:operation` 형식의 값이 있는 `name` 문자열 필드가 포함된 권한 오브젝트 배열입니다. 예를 들어 W&B Runs에 대한 삭제 작업에 대한 권한 오브젝트의 `name`은 `run:delete`입니다. |

-   **요청 예시**:

```bash
PATCH /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add", // 작업 유형을 나타냅니다. 다른 가능한 값은 `remove`입니다.
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

### 사용자 지정 역할 메타데이터 업데이트

-   **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
-   **메서드**: PUT
-   **설명**: W&B organization에서 사용자 지정 역할의 이름, 설명 또는 상속된 역할을 업데이트합니다. 이 작업은 사용자 지정 역할의 기존 비상속 사용자 지정 권한에 영향을 미치지 않습니다.
-   **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| name | 문자열 | 사용자 지정 역할의 이름 |
| description | 문자열 | 사용자 지정 역할에 대한 설명 |
| inheritedFrom | 문자열 | 사용자 지정 역할이 상속할 미리 정의된 역할입니다. `member` 또는 `viewer`일 수 있습니다. |

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