---
displayed_sidebar: default
---

# SCIM

System for Cross-domain Identity Management (SCIM) API는 인스턴스 또는 조직 관리자가 W&B 조직에서 사용자, 그룹 및 사용자 정의 역할을 관리할 수 있도록 합니다. SCIM 그룹은 W&B 팀에 매핑됩니다.

SCIM API는 `<host-url>/scim/`에서 접근 가능하며, [RC7643 프로토콜](https://www.rfc-editor.org/rfc/rfc7643)에 있는 필드의 서브셋을 지원하는 `/Users` 및 `/Groups` 엔드포인트를 지원합니다. 또한 공식 SCIM 스키마에는 포함되어 있지 않은 `/Roles` 엔드포인트를 포함합니다. W&B는 W&B 조직에서 사용자 정의 역할의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가합니다.

:::info
SCIM API는 전용 클라우드, 자체 관리 배포 및 SaaS 클라우드를 포함한 모든 호스팅 옵션에 적용됩니다. SaaS 클라우드에서는 조직 관리자가 SCIM API 요청이 올바른 조직으로 가도록 사용자 설정에서 기본 조직을 구성해야 합니다. 설정은 `SCIM API 조직` 섹션에서 사용할 수 있습니다.
:::

## 인증

SCIM API는 인스턴스 또는 조직 관리자가 기본 인증을 사용하여 자신의 API 키로 접근할 수 있습니다. 기본 인증을 사용할 때 `username:password`에 대한 base64 인코딩 문자열을 공백과 함께 `Basic`이라는 단어 다음에 포함하는 `Authorization` 헤더와 함께 HTTP 요청을 보냅니다. 예를 들어, `demo:p@55w0rd`로 인증하려면 헤더는 `Authorization: Basic ZGVtbzpwQDU1dzByZA==`여야 합니다.

## 사용자 리소스

SCIM 사용자 리소스는 W&B 사용자에 매핑됩니다.

### 사용자 조회

- **엔드포인트:** **`<host-url>/scim/Users/{id}`**
- **메소드**: GET
- **설명**: 사용자의 고유 ID를 제공하여 사용자 정보를 검색합니다.
- **요청 예시**:

```bash
GET /scim/Users/abc
```

- **응답 예시**:

```bash
(Status 200)
```

```json
{
    "active": true,
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

### 사용자 목록 조회

- **엔드포인트:** **`<host-url>/scim/Users`**
- **메소드**: GET
- **설명**: 사용자 목록을 검색합니다.
- **요청 예시**:

```bash
GET /scim/Users
```

- **응답 예시**:

```bash
(Status 200)
```

```json
{
    "Resources": [
        {
            "active": true,
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

| 필드 | 유형 | 필수 여부 |
| --- | --- | --- |
| emails | 다중 값 배열 | 예 (‘primary’ 이메일 설정 필수) |
| userName | 문자열 | 예 |
- **요청 예시**:

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

- **응답 예시**:

```bash
(Status 201)
```

```json
{
    "active": true,
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

### 사용자 비활성화

- **엔드포인트**: **`<host-url>/scim/Users/{id}`**
- **메소드**: DELETE
- **설명**: 사용자의 고유 ID를 제공하여 사용자를 비활성화합니다.
- **요청 예시**:

```bash
DELETE /scim/Users/abc
```

- **응답 예시**:

```json
(Status 204)
```

### 사용자 재활성화

- SCIM API에서는 이전에 비활성화된 사용자를 현재 지원하지 않습니다.

## 그룹 리소스

SCIM 그룹 리소스는 W&B 팀에 매핑됩니다. 즉, W&B 배포에서 SCIM 그룹을 생성하면 W&B 팀이 생성됩니다. 다른 그룹 엔드포인트에도 동일하게 적용됩니다.

### 팀 조회

- **엔드포인트**: **`<host-url>/scim/Groups/{id}`**
- **메소드**: GET
- **설명**: 팀의 고유 ID를 제공하여 팀 정보를 검색합니다.
- **요청 예시**:

```bash
GET /scim/Groups/ghi
```

- **응답 예시**:

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

### 팀 목록 조회

- **엔드포인트**: **`<host-url>/scim/Groups`**
- **메소드**: GET
- **설명**: 팀 목록을 검색합니다.
- **요청 예시**:

```bash
GET /scim/Groups
```

- **응답 예시**:

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
- **설명**: 새 팀 리소스를 생성합니다.
- **지원되는 필드**:

| 필드 | 유형 | 필수 여부 |
| --- | --- | --- |
| displayName | 문자열 | 예 |
| members | 다중 값 배열 | 예 (`value` 하위 필드 필요, 사용자 ID에 매핑됨) |
- **요청 예시**:

`wandb-support`라는 팀을 `dev-user2`가 회원인 상태로 생성합니다.

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

- **응답 예시**:

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
- **지원되는 작업**: 멤버 `추가`, 멤버 `제거`
- **요청 예시**:

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

- **응답 예시**:

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

- SCIM API에서는 팀에 연결된 추가 데이터가 있기 때문에 팀을 삭제하는 것을 현재 지원하지 않습니다. 앱에서 팀을 삭제하여 모든 것을 삭제하고 싶은지 확인하세요.

## 역할 리소스

SCIM 역할 리소스는 W&B 사용자 정의 역할에 매핑됩니다. 앞서 언급한 바와 같이 `/Roles` 엔드포인트는 공식 SCIM 스키마의 일부가 아닙니다. W&B는 W&B 조직에서 사용자 정의 역할의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가합니다.

### 사용자 정의 역할 조회

- **엔드포인트:** **`<host-url>/scim/Roles/{id}`**
- **메소드**: GET
- **설명**: 역할의 고유 ID를 제공하여 사용자 정의 역할 정보를 검색합니다.
- **요청 예시**:

```bash
GET /scim/Roles/abc
```

- **응답 예시**:

```bash
(Status 200)
```

```json
{
    "description": "예시를 위한 샘플 사용자 정의 역할",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 사전 정의된 역할을 나타냅니다.
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "샘플 사용자 정의 역할",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // member 사전 정의된 역할에서 상속됨
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 관리자가 추가한 사용자 정의 권한
        }
    ],
    "schemas": [
        ""
    ]
}
```

### 사용자 정의 역할 목록 조회

- **엔드포인트:** **`<host-url>/scim/Roles`**
- **메소드**: GET
- **설명**: W&B 조직의 모든 사용자 정의 역할 정보를 검색합니다.
- **요청 예시**:

```bash
GET /scim/Roles
```

- **응답 예시**:

```bash
(Status 200)
```

```json
{
   "Resources": [
        {
            "description": "예시를 위한 샘플 사용자 정의 역할",
            "id": "Um9sZTo3",
            "inheritedFrom": "member", // 사용자 정의 역할이 상속하는 사전 정의된 역할을 나타냅니다.
            "meta": {
                "resourceType": "Role",
                "created": "2023-11-20T23:10:14Z",
                "lastModified": "2023-11-20T23:31:23Z",
                "location": "Roles/Um9sZTo3"
            },
            "name": "샘플 사용자 정의 역할",
            "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
            "permissions": [
                {
                    "name": "artifact:read",
                    "isInherited": true // member 사전 정의된 역할에서 상속됨
                },
                ...
                ...
                {
                    "name": "project:update",
                    "isInherited": false // 관리자가 추가한 사용자 정의 권한
                }
            ],
            "schemas": [
                ""
            ]
        },
        {
            "description": "또 다

### 사용자 정의 역할 메타데이터 업데이트

- **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
- **메소드**: PUT
- **설명**: W&B 조직의 사용자 정의 역할의 이름, 설명 또는 상속받은 역할을 업데이트합니다. 이 작업은 기존의, 즉 상속받지 않은 사용자 정의 권한에는 영향을 주지 않습니다.
- **지원되는 필드**:

| 필드 | 타입 | 필수 |
| --- | --- | --- |
| name | String | 사용자 정의 역할의 이름 |
| description | String | 사용자 정의 역할의 설명 |
| inheritedFrom | String | 사용자 정의 역할이 상속받은 사전 정의된 역할입니다. `member` 또는 `viewer`일 수 있습니다. |
- **요청 예시**:

```bash
PUT /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
    "name": "Sample custom role",
    "description": "예제를 위한 샘플 사용자 정의 역할이지만 이제 viewer를 기반으로 합니다",
    "inheritedFrom": "viewer"
}
```

- **응답 예시**:

```bash
(Status 200)
```

```json
{
    "description": "예제를 위한 샘플 사용자 정의 역할이지만 이제 viewer를 기반으로 합니다", // 요청에 따라 설명 변경됨
    "id": "Um9sZTo3",
    "inheritedFrom": "viewer", // 요청에 따라 변경된 사전 정의된 역할을 나타냅니다
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
            "isInherited": true // viewer 사전 정의된 역할로부터 상속받음
        },
        ... // 업데이트 후에 viewer에는 있지만 member에는 없는 모든 권한은 상속받지 않습니다
        {
            "name": "project:update",
            "isInherited": false // 관리자가 추가한 사용자 정의 권한
        },
        {
            "name": "project:delete",
            "isInherited": false // 관리자가 추가한 사용자 정의 권한
        }
    ],
    "schemas": [
        ""
    ]
}
```

### 사용자에게 조직 수준 역할 할당

- **엔드포인트**: **`<host-url>/scim/Roles/{userId}`**
- **메소드**: PATCH
- **설명**: 사용자에게 조직 수준의 역할을 할당합니다. 역할은 [여기](./manage-users#invite-users)에 설명된 `admin`, `viewer`, `member` 중 하나일 수 있습니다. Public Cloud의 경우 사용자 설정에서 SCIM API에 대한 올바른 조직이 구성되어 있는지 확인하세요.
- **지원되는 필드**:

| 필드 | 타입 | 필수 |
| --- | --- | --- |
| op | String | 작업 유형입니다. 허용되는 값은 `replace`입니다. |
| path | String | 역할 할당 작업이 적용되는 범위입니다. 허용되는 값은 `organizationRole`입니다. |
| value | String | 사용자에게 할당할 사전 정의된 조직 수준 역할입니다. `admin`, `viewer`, `member` 중 하나일 수 있습니다. 이 필드는 사전 정의된 역할에 대해 대소문자를 구분하지 않습니다. |
- **요청 예시**:

:::caution
역할 할당 API에 대한 요청 경로는 사용자 정의 역할 API, 특히 [PATCH - 사용자 정의 역할 권한 업데이트](#update-custom-role-permissions) 작업과 동일합니다. 차이점은 역할 할당 API의 URI가 `:userId` 파라미터를 기대하는 반면, 사용자 정의 역할 API의 URI는 `:roleId`를 기대합니다. 두 API에 대한 예상 요청 본문도 다릅니다.

URI의 파라미터 값과 요청 본문이 의도한 작업에 맞도록 주의하세요.
:::

```bash
PUT /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "organizationRole",
            "value": "admin" // 사용자의 조직 범위 역할을 admin으로 설정합니다
        }
    ]
}
```

- **응답 예시**:
이는 [사용자 리소스](#user-resource)의 경우와 같이 User 오브젝트를 반환합니다.

```bash
(Status 200)
```

```json
{
    "active": true,
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
    "teamRoles": [  // 사용자가 속한 모든 팀의 역할을 반환합니다
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 사용자의 조직 범위 역할을 반환합니다
}
```

### 사용자에게 팀 수준 역할 할당

- **엔드포인트**: **`<host-url>/scim/Roles/{userId}`**
- **메소드**: PATCH
- **설명**: 사용자에게 팀 수준의 역할을 할당합니다. 역할은 [여기](./manage-users#team-roles)에 설명된 `admin`, `viewer`, `member` 또는 사용자 정의 역할 중 하나일 수 있습니다. Public Cloud의 경우 사용자 설정에서 SCIM API에 대한 올바른 조직이 구성되어 있는지 확인하세요.
- **지원되는 필드**:

| 필드 | 타입 | 필수 |
| --- | --- | --- |
| op | String | 작업 유형입니다. 허용되는 값은 `replace`입니다. |
| path | String | 역할 할당 작업이 적용되는 범위입니다. 허용되는 값은 `teamRoles`입니다. |
| value | 오브젝트 배열 | 오브젝트로 구성된 한 오브젝트 배열입니다. 오브젝트는 `teamName`과 `roleName` 속성으로 구성됩니다. `teamName`은 사용자가 역할을 가진 팀의 이름이고, `roleName`은 `admin`, `viewer`, `member` 또는 사용자 정의 역할 중 하나일 수 있습니다. 이 필드는 사전 정의된 역할에 대해 대소문자를 구분하지 않으며 사용자 정의 역할에 대해서는 대소문자를 구분합니다. |
- **요청 예시**:

:::caution
역할 할당 API에 대한 요청 경로는 사용자 정의 역할 API, 특히 [PATCH - 사용자 정의 역할 권한 업데이트](#update-custom-role-permissions) 작업과 동일합니다. 차이점은 역할 할당 API의 URI가 `:userId` 파라미터를 기대하는 반면, 사용자 정의 역할 API의 URI는 `:roleId`를 기대합니다. 두 API에 대한 예상 요청 본문도 다릅니다.

URI의 파라미터 값과 요청 본문이 의도한 작업에 맞도록 주의하세요.
:::

```bash
PUT /scim/Roles/abc
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
                    "roleName": "admin", // 역할 이름은 사전 정의된 역할에 대해 대소문자를 구분하지 않고 사용자 정의 역할에 대해서는 대소문자를 구분합니다
                    "teamName": "team1" // team1 팀에서 사용자의 역할을 admin으로 설정합니다
                }
            ]
        }
    ]
}
```

- **응답 예시**:
이는 [사용자 리소스](#user-resource)의 경우와 같이 User 오브젝트를 반환합니다.

```bash
(Status 200)
```

```json
{
    "active": true,
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
    "teamRoles": [  // 사용자가 속한 모든 팀의 역할을 반환합니다
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 사용자의 조직 범위 역할을 반환합니다
}
```