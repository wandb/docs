---
title: SCIM 을 사용하여 사용자, 그룹 및 역할 관리
menu:
  default:
    identifier: ko-guides-hosting-iam-scim
    parent: identity-and-access-management-iam
weight: 4
---

{{% alert %}}
[SCIM이 실제로 동작하는 영상 보기](https://www.youtube.com/watch?v=Nw3QBqV0I-o) (12분)
{{% /alert %}}

## 개요

SCIM(System for Cross-domain Identity Management) API는 인스턴스 또는 조직 관리자가 W&B 조직 내의 사용자, 그룹, 커스텀 역할을 관리할 수 있도록 합니다. SCIM 그룹은 W&B 팀과 매핑됩니다.

SCIM API는 `<host-url>/scim/`에서 이용 가능하며, `/Users` 및 `/Groups` 엔드포인트를 [RC7643 프로토콜](https://www.rfc-editor.org/rfc/rfc7643)에 정의된 필드의 일부 서브셋으로 지원합니다. 추가로 공식 SCIM 스키마에서는 다루지 않지만, W&B의 커스텀 역할 자동 관리를 위해 `/Roles` 엔드포인트도 제공합니다.

{{% alert %}}
여러 엔터프라이즈 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 조직의 관리자인 경우 SCIM API 요청을 보낼 조직을 지정해야 합니다. 프로필 이미지를 클릭한 후 **User Settings**(사용자 설정)을 클릭하세요. 해당 설정은 **Default API organization**(기본 API 조직)이라는 이름입니다. 이 설정은 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}), [셀프 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}), [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 등 모든 호스팅 옵션에서 필요합니다. SaaS Cloud의 경우, 조직 관리자가 반드시 사용자 설정에서 기본 조직을 지정하여 SCIM API 요청이 올바른 조직으로 전송되도록 해야 합니다.

선택한 호스팅 옵션에 따라 본 페이지의 예시에서 사용하는 `<host-url>` 값이 결정됩니다.

또한, 예시에서는 `abc`, `def`와 같은 사용자 ID를 사용하지만 실제 요청 및 응답에서는 해시된 사용자 ID가 제공됩니다.
{{% /alert %}}

## 인증

SCIM API에 엑세스하려면 아래 두 가지 방법으로 인증할 수 있습니다.

### 사용자

조직 또는 인스턴스 관리자는 본인의 API 키를 활용해 기본 인증 방식으로 SCIM API에 접근할 수 있습니다. HTTP 요청의 `Authorization` 헤더를 `Basic` 뒤에 한 칸 띄우고, `username:API-KEY` 형식의 문자열을 Base64로 인코딩한 값을 입력하세요. 즉, 본인의 username과 API 키를 `:`로 결합한 후 Base64 인코딩하면 됩니다. 예를 들어 `demo:p@55w0rd`로 인증하려면 헤더는 다음과 같이 설정됩니다: `Authorization: Basic ZGVtbzpwQDU1dzByZA==`

### 서비스 계정

`admin` 역할을 가진 조직 서비스 계정도 SCIM API에 접근할 수 있습니다. 이때 username은 비워 두고, 오직 API 키만 사용합니다. 서비스 계정의 API 키는 조직 대시보드의 **Service account** 탭에서 확인할 수 있습니다. 자세한 내용은 [조직 범위 서비스 계정]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md/#organization-scoped-service-accounts" lang="ko" >}})을 참고하세요.

HTTP 요청의 `Authorization` 헤더는 `Basic` 뒤에 한 칸을 띄우고, `:API-KEY` 형식의 문자열을 Base64로 인코딩한 값을 사용합니다(맨 앞에 `:`를 남겨둡니다). 예를 들어 API 키가 `sa-p@55w0rd`인 경우: `Authorization: Basic OnNhLXBANTV3MHJk`

## 사용자 관리

SCIM 사용자 리소스는 W&B 사용자와 매핑됩니다. 아래 엔드포인트를 이용해 조직 내 사용자를 관리할 수 있습니다.

### 사용자 조회

조직 내 특정 사용자의 정보를 조회합니다.

#### 엔드포인트
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: GET

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|-----------|------|----------|-------------|
| id | string | 예 | 사용자의 고유 ID |

#### 예시

{{< tabpane text=true >}}
{{% tab header="Get User Request" %}}
```bash
GET /scim/Users/abc
```
{{% /tab %}}
{{% tab header="Get User Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

### 사용자 목록 조회

조직 내 모든 사용자 목록을 조회합니다.

#### 엔드포인트
- **URL**: `<host-url>/scim/Users`
- **Method**: GET

#### 예시

{{< tabpane text=true >}}
{{% tab header="List Users Request" %}}
```bash
GET /scim/Users
```
{{% /tab %}}
{{% tab header="List Users Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

### 사용자 생성

조직 내에 새로운 사용자를 생성합니다.

#### 엔드포인트
- **URL**: `<host-url>/scim/Users`
- **Method**: POST

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|-----------|------|----------|-------------|
| emails | array | 예 | 이메일 객체의 배열. 반드시 기본(primary) 이메일이 포함되어야 함 |
| userName | string | 예 | 신규 사용자의 username |

#### 예시

{{< tabpane text=true >}}
{{% tab header="Create User Request (Dedicated/Self-managed)" %}}
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
            "value": "dev-user2@test.com"
        }
    ],
    "userName": "dev-user2"
}
```
{{% /tab %}}
{{% tab header="Create User Request (Multi-tenant)" %}}
```bash
POST /scim/Users
```

```json
{
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User",
        "urn:ietf:params:scim:schemas:extension:teams:2.0:User"
    ],
    "emails": [
        {
            "primary": true,
            "value": "dev-user2@test.com"
        }
    ],
    "userName": "dev-user2",
    "urn:ietf:params:scim:schemas:extension:teams:2.0:User": {
        "teams": ["my-team"]
    }
}
```
{{% /tab %}}
{{< /tabpane >}}

#### 응답

{{< tabpane text=true >}}
{{% tab header="Create User Response (Dedicated/Self-managed)" %}}
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
{{% /tab %}}
{{% tab header="Create User Response (Multi-tenant)" %}}
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
        "urn:ietf:params:scim:schemas:core:2.0:User",
        "urn:ietf:params:scim:schemas:extension:teams:2.0:User"
    ],
    "userName": "dev-user2",
    "organizationRole": "member",
    "teamRoles": [
        {
            "teamName": "my-team",
            "roleName": "member"
        }
    ],
    "groups": [
        {
            "value": "my-team-id"
        }
    ]
}
```
{{% /tab %}}
{{< /tabpane >}}

### 사용자 삭제

{{% alert color="warning" title="관리자 엑세스 유지" %}}
항상 인스턴스 또는 조직에 최소 한 명 이상의 관리자 사용자가 있어야 합니다. 그렇지 않으면 어떤 사용자도 W&B 조직 계정의 설정이나 유지 관리를 할 수 없습니다. 만약 SCIM 또는 다른 자동화 프로세스를 통해 사용자를 비활성화할 때, 시스템 마지막 관리자가 실수로 제거될 수 있습니다.

운영 프로세스 개발 지원이나 관리자 엑세스를 복구하려면 [support](mailto:support@wandb.com)에 문의하세요.
{{% /alert %}}

조직에서 사용자를 완전히 삭제합니다.

#### 엔드포인트
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: DELETE

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|-----------|------|----------|-------------|
| id | string | 예 | 삭제할 사용자의 고유 ID |

#### 예시

{{< tabpane text=true >}}
{{% tab header="Delete User Request" %}}
```bash
DELETE /scim/Users/abc
```
{{% /tab %}}
{{% tab header="Delete User Response" %}}
```bash
(Status 204)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
사용자를 일시적으로 비활성화하려면 `PATCH` 엔드포인트를 이용하는 [사용자 비활성화](#deactivate-user) API를 참고하세요.
{{% /alert %}}

### 사용자 비활성화

조직 내 사용자를 일시적으로 비활성화합니다.

#### 엔드포인트
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|-----------|------|----------|-------------|
| id | string | 예 | 비활성화할 사용자의 고유 ID |
| op | string | 예 | "replace"여야 함 |
| value | object | 예 | `{"active": false}` 값을 가진 오브젝트 |

{{% alert %}}
[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서는 사용자 비활성화 및 재활성화 작업이 지원되지 않습니다.
{{% /alert %}}

#### 예시

{{< tabpane text=true >}}
{{% tab header="Deactivate User Request" %}}
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
{{% /tab %}}
{{% tab header="Deactivate User Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

### 사용자 재활성화

조직 내에서 이전에 비활성화된 사용자를 다시 활성화합니다.

#### 엔드포인트
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|-----------|------|----------|-------------|
| id | string | 예 | 재활성화할 사용자의 고유 ID |
| op | string | 예 | "replace"여야 함 |
| value | object | 예 | `{"active": true}` 값을 가진 오브젝트 |

{{% alert %}}
[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서는 사용자 비활성화 및 재활성화 작업이 지원되지 않습니다.
{{% /alert %}}

#### 예시

{{< tabpane text=true >}}
{{% tab header="Reactivate User Request" %}}
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
{{% /tab %}}
{{% tab header="Reactivate User Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

### 조직 역할 할당

사용자에게 조직 수준 역할을 할당합니다.

#### 엔드포인트
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|-----------|------|----------|-------------|
| id | string | 예 | 사용자의 고유 ID |
| op | string | 예 | "replace"여야 함 |
| path | string | 예 | "organizationRole"여야 함 |
| value | string | 예 | 역할 이름 ("admin" 또는 "member") |

{{% alert %}}
`viewer` 역할은 더 이상 지원하지 않으며, UI에서 설정할 수 없습니다. SCIM을 통해 `viewer` 역할을 할당하려 하면 W&B는 자동으로 해당 사용자를 `member` 역할로 지정합니다. 사용자가 가능한 경우, Models와 Weave 시트가 자동 할당됩니다. 아니면 `Seat limit reached` 오류가 기록됩니다. **Registry**를 사용하는 조직에서는, 사용자가 조직 레벨에서 보이는 registry에 자동으로 `viewer` 역할을 받게 됩니다.
{{% /alert %}}

#### 예시

{{< tabpane text=true >}}
{{% tab header="Assign Org Role Request" %}}
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
            "value": "admin"
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Assign Org Role Response" %}}
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
    "teamRoles": [
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin"
}
```
{{% /tab %}}
{{< /tabpane >}}

### 팀 역할 할당

사용자에게 팀 수준의 역할을 할당합니다.

#### 엔드포인트
- **URL**: `<host-url>/scim/Users/{id}`
- **Method**: PATCH

#### 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|-----------|------|----------|-------------|
| id | string | 예 | 사용자의 고유 ID |
| op | string | 예 | "replace"여야 함 |
| path | string | 예 | "teamRoles"여야 함 |
| value | array | 예 | `teamName`과 `roleName`을 포함한 객체의 배열 |

#### 예시

{{< tabpane text=true >}}
{{% tab header="Assign Team Role Request" %}}
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
                    "roleName": "admin",
                    "teamName": "team1"
                }
            ]
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Assign Team Role Response" %}}
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
    "teamRoles": [
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin"
}
```
{{% /tab %}}
{{< /tabpane >}}

## 그룹 리소스

SCIM 그룹 리소스는 W&B 팀에 매핑됩니다. 즉, SCIM 그룹을 W&B 배포에 생성하면 새로운 W&B 팀이 생성됩니다. 다른 그룹 관련 엔드포인트도 동일하게 동작합니다.

### 팀 조회

- **엔드포인트**: **`<host-url>/scim/Groups/{id}`**
- **Method**: GET
- **설명**: 팀의 고유 ID를 제공하여 팀 정보를 조회합니다.
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
- **Method**: GET
- **설명**: 팀 목록을 조회합니다.
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
- **Method**: POST
- **설명**: 새로운 팀 리소스를 생성합니다.
- **지원 필드**:

| 필드 | 타입 | 필수 |
| --- | --- | --- |
| displayName | String | 예 |
| members | Multi-Valued Array | 예 (`value` 서브필드는 필수이며 user ID와 매핑됨) |
- **요청 예시**:

`dev-user2` 사용자를 멤버로 가지는 `wandb-support` 팀 생성 예시입니다.

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
- **Method**: PATCH
- **설명**: 기존 팀 멤버십 목록을 업데이트합니다.
- **지원 가능 작업**: 멤버 추가(`add`), 멤버 삭제(`remove`)

{{% alert %}}
멤버 삭제 작업은 RFC 7644 SCIM 프로토콜 명세를 따릅니다. 특정 사용자를 삭제하려면 `members[value eq "{user_id}"]` 형식의 필터를, 모든 사용자를 삭제하려면 `members`를 사용하면 됩니다.
{{% /alert %}}

{{% alert color="info" %}}
요청 시 `{team_id}`와 `{user_id}`는 실제 팀 ID와 사용자 ID로 교체해 사용하세요.
{{% /alert %}}

**팀에 사용자 추가하기**

`wandb-devs`에 `dev-user2`를 추가하는 예시:

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Groups/{team_id}
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
                    "value": "{user_id}"
                }
            ]
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Response" %}}
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
{{% /tab %}}
{{< /tabpane >}}

**팀에서 특정 사용자 제거하기**

`wandb-devs`에서 `dev-user2`를 제거하는 예시:

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Groups/{team_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "remove",
            "path": "members[value eq \"{user_id}\"]"
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Response" %}}
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
        "lastModified": "2023-10-01T00:01:00Z",
        "location": "Groups/ghi"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:Group"
    ]
}
```
{{% /tab %}}
{{< /tabpane >}}

**팀에서 모든 사용자 제거하기**

`wandb-devs`에서 모든 사용자를 제거하는 예시:

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Groups/{team_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "remove",
            "path": "members"
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Response" %}}
```bash
(Status 200)
```

```json
{
    "displayName": "wandb-devs",
    "id": "ghi",
    "members": null,
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
{{% /tab %}}
{{< /tabpane >}}

### 팀 삭제

- SCIM API에서는 현재 팀 삭제가 지원되지 않습니다. 팀에 연결된 추가 데이터가 있기 때문입니다. 모든 데이터를 삭제하려면 앱 내에서 팀 삭제를 진행하세요.

## 역할 리소스

SCIM 역할 리소스는 W&B의 커스텀 역할과 매핑됩니다. 앞서 설명한 것처럼 `/Roles` 엔드포인트는 공식 SCIM 스키마의 일부는 아니지만, W&B 내 커스텀 역할의 자동 관리를 지원하기 위해 제공됩니다.

### 커스텀 역할 조회

- **엔드포인트:** **`<host-url>/scim/Roles/{id}`**
- **Method**: GET
- **설명**: 고유 ID를 입력하여 커스텀 역할 정보 조회
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
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 사전 정의된 역할임을 표시
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
            "isInherited": true // member 역할로부터 상속됨
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

### 커스텀 역할 목록 조회

- **엔드포인트:** **`<host-url>/scim/Roles`**
- **Method**: GET
- **설명**: W&B 조직 내 모든 커스텀 역할 정보 조회
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
            "description": "A sample custom role for example",
            "id": "Um9sZTo3",
            "inheritedFrom": "member", // 멤버 역할에서 상속됨
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
                    "isInherited": true // member 역할로부터 상속됨
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
            "inheritedFrom": "viewer", // viewer 역할에서 상속됨
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
                    "isInherited": true // viewer 역할로부터 상속됨
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
- **Method**: POST
- **설명**: W&B 조직 내 새로운 커스텀 역할 생성
- **지원 필드**:

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| name | String | 커스텀 역할의 이름 |
| description | String | 커스텀 역할 설명 |
| permissions | Object array | 각 permission 객체는 `name` 필드가 있어야 하며, 값은 `w&bobject:operation` 형식이어야 함. 예: run 삭제 권한의 경우 `run:delete` |
| inheritedFrom | String | 어떤 사전 정의 역할을 상속받을지 (예: `member` 또는 `viewer`) |
- **요청 예시**:

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

- **응답 예시**:

```bash
(Status 201)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 사전 정의된 역할임을 표시
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
            "isInherited": true // member 역할로부터 상속됨
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
- **Method**: DELETE
- **설명**: W&B 조직 내 커스텀 역할을 삭제합니다. **신중히 사용하십시오**. 커스텀 역할을 할당받았던 모든 사용자는, 이제 해당 커스텀 역할이 상속받았던 사전 정의 역할을 받게 됩니다.
- **요청 예시**:

```bash
DELETE /scim/Roles/abc
```
