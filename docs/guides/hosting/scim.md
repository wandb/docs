---
displayed_sidebar: default
---

# SCIM

System for Cross-domain Identity Management (SCIM) API는 인스턴스 또는 조직 관리자가 W&B 조직의 사용자, 그룹 및 커스텀 역할을 관리할 수 있게 해줍니다. SCIM 그룹은 W&B 팀에 매핑됩니다.

SCIM API는 `<host-url>/scim/`에서 액세스할 수 있으며, [RC7643 프로토콜](https://www.rfc-editor.org/rfc/rfc7643)에서 찾을 수 있는 필드의 서브세트와 함께 `/Users` 및 `/Groups` 엔드포인트를 지원합니다. 추가로 공식 SCIM 스키마에 포함되지 않은 `/Roles` 엔드포인트를 포함합니다. W&B는 W&B 조직에서 커스텀 역할의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가합니다.

:::info
SCIM API는 데디케이티드 클라우드, 자체 관리 배포 및 SaaS 클라우드를 포함한 모든 호스팅 옵션에 적용됩니다. SaaS 클라우드에서 조직 관리자는 SCIM API 요청이 올바른 조직으로 가도록 사용자 설정에서 기본 조직을 구성해야 합니다. 설정은 `SCIM API Organization` 섹션에서 사용할 수 있습니다.
:::

## 인증

SCIM API는 인스턴스 또는 조직 관리자가 API 키를 사용한 기본 인증으로 액세스할 수 있습니다. 기본 인증으로는 `username:password`를 위한 base64 인코딩 문자열을 포함하는 `Authorization` 헤더와 함께 HTTP 요청을 보냅니다. 여기서 `password`는 API 키입니다. 예를 들어, `demo:p@55w0rd`로 인증하려면 헤더는 `Authorization: Basic ZGVtbzpwQDU1dzByZA==`가 되어야 합니다.

## 사용자 리소스

SCIM 사용자 리소스는 W&B 사용자에 매핑됩니다.

### 사용자 가져오기

- **엔드포인트:** **`<host-url>/scim/Users/{id}`**
- **메서드**: GET
- **설명**: 사용자의 고유 ID를 제공하여 사용자 정보를 검색합니다.
- **요청 예**:

```bash
GET /scim/Users/abc
```

- **응답 예**:

```bash
(상태 200)
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

### 사용자 목록

- **엔드포인트:** **`<host-url>/scim/Users`**
- **메서드**: GET
- **설명**: 사용자 목록을 검색합니다.
- **요청 예**:

```bash
GET /scim/Users
```

- **응답 예**:

```bash
(상태 200)
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
- **메서드**: POST
- **설명**: 새 사용자 리소스를 생성합니다.
- **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| emails | 다중 값 배열 | 예 (`primary` 이메일이 설정되어야 함) |
| userName | 문자열 | 예 |
- **요청 예**:

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

- **응답 예**:

```bash
(상태 201)
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
- **메서드**: DELETE
- **설명**: 사용자의 고유 ID를 제공하여 사용자를 비활성화합니다.
- **요청 예**:

```bash
DELETE /scim/Users/abc
```

- **응답 예**:

```json
(상태 204)
```

### 사용자 재활성화

- SCIM API에서는 현재 비활성화된 사용자를 재활성화하는 것을 지원하지 않습니다.

## 그룹 리소스

SCIM 그룹 리소스는 W&B 팀에 매핑됩니다. 즉, W&B 배포에서 SCIM 그룹을 생성하면 W&B 팀이 생성됩니다. 다른 그룹 엔드포인트에도 동일하게 적용됩니다.

### 팀 가져오기

- **엔드포인트**: **`<host-url>/scim/Groups/{id}`**
- **메서드**: GET
- **설명**: 팀의 고유 ID를 제공하여 팀 정보를 검색합니다.
- **요청 예**:

```bash
GET /scim/Groups/ghi
```

- **응답 예**:

```bash
(상태 200)
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

- **엔드포인트**: **`<host-url>/scim/Groups`**
- **메서드**: GET
- **설명**: 팀 목록을 검색합니다.
- **요청 예**:

```bash
GET /scim/Groups
```

- **응답 예**:

```bash
(상태 200)
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
- **메서드**: POST
- **설명**: 새 팀 리소스를 생성합니다.
- **지원되는 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| displayName | 문자열 | 예 |
| members | 다중 값 배열 | 예 (`value` 하위 필드가 필요하며 사용자 ID에 매핑됨) |
- **요청 예**:

`wandb-support`라는 이름의 팀을 `dev-user2`를 멤버로 하여 생성합니다.

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

- **응답 예**:

```bash
(상태 201)
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
- **메서드**: PATCH
- **설명**: 기존 팀의 멤버십 목록을 업데이트합니다.
- **지원되는 작업**: 멤버 `추가`, 멤버 `제거`
- **요청 예**:

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

- **응답 예**:

```bash
(상태 200)
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

- SCIM API에서는 팀에 연결된 추가 데이터가 있기 때문에 팀을 삭제하는 것을 현재 지원하지 않습니다. 모든 것을 삭제하고 싶다면 앱에서 팀을 삭제하세요.

## 역할 리소스

SCIM 역할 리소스는 W&B 커스텀 역할에 매핑됩니다. 앞서 언급했듯이, `/Roles` 엔드포인트는 공식 SCIM 스키마의 일부가 아닙니다. W&B는 W&B 조직에서 커스텀 역할의 자동 관리를 지원하기 위해 `/Roles` 엔드포인트를 추가합니다.

### 커스텀 역할 가져오기

- **엔드포인트:** **`<host-url>/scim/Roles/{id}`**
- **메서드**: GET
- **설명**: 역할의 고유 ID를 제공하여 커스텀 역할에 대한 정보를 검색합니다.
- **요청 예**:

```bash
GET /scim/Roles/abc
```

- **응답 예**:

```bash
(상태 200)
```

```json
{
    "description": "예시를 위한 샘플 커스텀 역할",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 미리 정의된 역할을 나타냅니다
    "meta": {
        "resourceType": "Role",
        "created": "2023-11-20T23:10:14Z",
        "lastModified": "2023-11-20T23:31:23Z",
        "location": "Roles/Um9sZTo3"
    },
    "name": "샘플 커스텀 역할",
    "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
    "permissions": [
        {
            "name": "artifact:read",
            "isInherited": true // 멤버 미리 정의된 역할로부터 상속됨
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 관리자가 추가한 커스텀 권한
        }
    ],
    "schemas": [
        ""
    ]
}
```

### 커스텀 역할 목록

- **엔드포인트:** **`<host-url>/scim/Roles`**
- **메서드**: GET
- **설명**: W&B 조직의 모든 커스텀 역할에 대한 정보를 검색합니다.
- **요청 예**:

```bash
GET /scim/Roles
```

- **응답 예**:

```bash
(상태 200)
```

```json
{
   "Resources": [
        {
            "description": "예시를 위한 샘플 커스텀 역할",
            "id": "Um9sZTo3",
            "inheritedFrom": "member", // 커스텀 역할이 상속받는 미리 정의된 역할을 나타냅니다
            "meta": {
                "resourceType": "Role",
                "created": "2023-11-20T23:10:14Z",
                "lastModified": "2023-11-20T23:31:23Z",
                "location": "Roles/Um9sZTo3"
            },
            "name": "샘플 커스텀 역할",
            "organizationID": "T3JnYW5pemF0aW9uOjE0ODQ1OA==",
            "permissions": [
                {
                    "name": "artifact:read",
                    "isInherited": true // 멤버 미리 정의된 역할로부터 상속됨
                },
                ...


### 사용자 정의 역할 생성

- **엔드포인트**: **`<host-url>/scim/Roles`**
- **메서드**: POST
- **설명**: W&B 조직에 새로운 사용자 정의 역할을 생성합니다.
- **지원 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| name | 문자열 | 사용자 정의 역할의 이름 |
| description | 문자열 | 사용자 정의 역할의 설명 |
| permissions | 객체 배열 | 각 객체가 `name` 문자열 필드를 포함하고 이 값은 `w&bobject:operation` 형식의 값을 가지는 권한 객체 배열. 예를 들어, W&B 실행에 대한 삭제 작업을 위한 권한 객체는 `name`을 `run:delete`로 가집니다. |
| inheritedFrom | 문자열 | 사용자 정의 역할이 상속받을 기본 역할. `member` 또는 `viewer`일 수 있습니다. |
- **요청 예시**:

```bash
POST /scim/Roles
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
    "name": "샘플 사용자 정의 역할",
    "description": "예시를 위한 샘플 사용자 정의 역할",
    "permissions": [
        {
            "name": "project:update"
        }
    ],
    "inheritedFrom": "member"
}
```

- **응답 예시**:

```bash
(Status 201)
```

```json
{
    "description": "예시를 위한 샘플 사용자 정의 역할",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 기본 역할을 나타냄
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
            "isInherited": true // member 기본 역할에서 상속됨
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

### 사용자 정의 역할 삭제

- **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
- **메서드**: DELETE
- **설명**: W&B 조직에서 사용자 정의 역할을 삭제합니다. **주의해서 사용하세요**. 작업을 수행하기 전에 사용자 정의 역할에 할당되었던 모든 사용자에게 이제 상속받은 기본 역할이 할당됩니다.
- **요청 예시**:

```bash
DELETE /scim/Roles/abc
```

- **응답 예시**:

```bash
(Status 204)
```

### 사용자 정의 역할 권한 업데이트

- **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
- **메서드**: PATCH
- **설명**: W&B 조직에서 사용자 정의 역할의 권한을 추가하거나 제거합니다.
- **지원 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| operations | 객체 배열 | 작업 객체 배열 |
| op | 문자열 | 작업 객체 내에서의 작업 유형. `add` 또는 `remove`일 수 있습니다. |
| path | 문자열 | 작업 객체에서 정적 필드. 허용되는 값은 `permissions`만 있습니다. |
| value | 객체 배열 | 각 객체가 `name` 문자열 필드를 포함하고 이 값은 `w&bobject:operation` 형식의 값을 가지는 권한 객체 배열. 예를 들어, W&B 실행에 대한 삭제 작업을 위한 권한 객체는 `name`을 `run:delete`로 가집니다. |
- **요청 예시**:

:::caution
사용자 정의 역할 권한 업데이트 API의 요청 경로는 역할 할당 API, 즉 [PATCH - 사용자에게 조직 수준 역할 할당](#assign-organization-level-role-to-user) 및 [PATCH - 사용자에게 팀 수준 역할 할당](#assign-team-level-role-to-user) 작업과 동일합니다. 차이점은 사용자 정의 역할 API의 URI가 `:roleId` 파라미터를 기대하는 반면, 역할 할당 API의 URI는 `:userId`를 기대합니다. 두 유형의 API 모두에 대한 예상 요청 본문도 다릅니다.

URI에서 파라미터 값과 요청 본문이 의도한 작업에 맞도록 주의하세요.
:::

```bash
PATCH /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add", // 작업 유형을 나타냄, 다른 가능한 값은 `remove`
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

- **응답 예시**:

```bash
(Status 200)
```

```json
{
    "description": "예시를 위한 샘플 사용자 정의 역할",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 기본 역할을 나타냄
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
            "isInherited": true // member 기본 역할에서 상속됨
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 업데이트 전에 관리자가 추가한 기존의 사용자 정의 권한
        },
        {
            "name": "project:delete",
            "isInherited": false // 업데이트의 일부로 관리자가 추가한 새로운 사용자 정의 권한
        }
    ],
    "schemas": [
        ""
    ]
}
```

### 사용자 정의 역할 메타데이터 업데이트

- **엔드포인트**: **`<host-url>/scim/Roles/{id}`**
- **메서드**: PUT
- **설명**: W&B 조직에서 사용자 정의 역할의 이름, 설명 또는 상속받은 역할을 업데이트합니다. 이 작업은 사용자 정의 역할의 기존(즉, 상속받지 않은) 사용자 정의 권한에는 영향을 미치지 않습니다.
- **지원 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| name | 문자열 | 사용자 정의 역할의 이름 |
| description | 문자열 | 사용자 정의 역할의 설명 |
| inheritedFrom | 문자열 | 사용자 정의 역할이 상속받는 기본 역할. `member` 또는 `viewer`일 수 있습니다. |
- **요청 예시**:

```bash
PUT /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
    "name": "샘플 사용자 정의 역할",
    "description": "이제 viewer를 기반으로 한 예시를 위한 샘플 사용자 정의 역할",
    "inheritedFrom": "viewer"
}
```

- **응답 예시**:

```bash
(Status 200)
```

```json
{
    "description": "이제 viewer를 기반으로 한 예시를 위한 샘플 사용자 정의 역할", // 요청에 따라 변경된 설명
    "id": "Um9sZTo3",
    "inheritedFrom": "viewer", // 요청에 따라 변경된 기본 역할을 나타냄
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
            "isInherited": true // viewer 기본 역할에서 상속됨
        },
        ... // 업데이트 후에 member 기본 역할에 있지만 viewer에는 없는 모든 권한은 상속받지 않음
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
- **메서드**: PATCH
- **설명**: 사용자에게 조직 수준의 역할을 할당합니다. 역할은 [여기](./manage-users#invite-users)에 설명된 `admin`, `viewer`, `member` 중 하나일 수 있습니다. 공용 클라우드의 경우, 사용자 설정에서 SCIM API를 위한 올바른 조직을 구성했는지 확인하세요.
- **지원 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. 허용되는 값은 `replace`만 있습니다. |
| path | 문자열 | 역할 할당 작업이 적용되는 범위. 허용되는 값은 `organizationRole`만 있습니다. |
| value | 문자열 | 사용자에게 할당할 기본 조직 수준 역할. `admin`, `viewer`, `member` 중 하나일 수 있습니다. 이 필드는 기본 역할에 대해 대소문자를 구분하지 않습니다. |
- **요청 예시**:

:::caution
역할 할당 API의 요청 경로는 사용자 정의 역할 API, 특히 [PATCH - 사용자 정의 역할 권한 업데이트](#update-custom-role-permissions) 작업과 동일합니다. 차이점은 역할 할당 API의 URI가 `:userId` 파라미터를 기대하는 반면, 사용자 정의 역할 API의 URI는 `:roleId`를 기대합니다. 두 API 모두에 대한 예상 요청 본문도 다릅니다.

URI에서 파라미터 값과 요청 본문이 의도한 작업에 맞도록 주의하세요.
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
            "value": "admin" // 사용자의 조직 범위 역할을 admin으로 설정함
        }
    ]
}
```

- **응답 예시**:
이는 [사용자 리소스](#user-resource)의 경우와 같이 사용자 객체를 반환합니다.

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
    "teamRoles": [  // 사용자가 속한 모든 팀에서의 사용자의 역할을 반환함
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 조직 범위에서 사용자의 역할을 반환함
}
```

### 사용자에게 팀 수준 역할 할당

- **엔드포인트**: **`<host-url>/scim/Roles/{userId}`**
- **메서드**: PATCH
- **설명**: 사용자에게 팀 수준의 역할을 할당합니다. 역할은 [여기](./manage-users#team-roles)에 설명된 `admin`, `viewer`, `member` 또는 사용자 정의 역할 중 하나일 수 있습니다. 공용 클라우드의 경우, 사용자 설정에서 SCIM API를 위한 올바른 조직을 구성했는지 확인하세요.
- **지원 필드**:

| 필드 | 유형 | 필수 |
| --- | --- | --- |
| op | 문자열 | 작업 유형. 허용되는 값은 `replace`만 있습니다. |
| path | 문자열 | 역할 할당 작업이 적용되는 범위. 허용되는 값은 `teamRoles`만 있습니다. |
| value | 객체 배열 | `teamName`과 `roleName` 속성을 가진 객체가 하나 포함된 배열. `teamName`은 사용자가 역할을 가지는 팀의 이름이며, `roleName`은 `admin`, `viewer`, `member` 또는 사용자 정의 역할 중 하나일 수 있습니다. 이 필드는 기본 역할에 대해 대소문자를 구분하지 않고 사용자 정의 역할에 대해 대소문자를 구분합니다. |
- **요청 예시**:

:::caution
역할 할당 API의 요청 경로는 사용자 정의 역할 API, 특히 [PATCH - 사용자 정의 역할 권한 업데이트](#update-custom-role-permissions) 작업과 동일합니다. 차이점은 역할 할당 API의 URI가 `:userId` 파라미터를 기대하는 반면, 사용자 정의 역할 API의 URI는 `:roleId`를 기대합니다. 두 API 모두에 대한 예상 요청 본문도 다릅니다.

URI에서 파라미터 값과 요청 본문이 의도한 작업에 맞도록 주의하세요.
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
            "value":