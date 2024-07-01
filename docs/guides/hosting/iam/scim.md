---
displayed_sidebar: default
---

# SCIM

System for Cross-domain Identity Management (SCIM) API は、インスタンスまたは組織の管理者が W&B 組織内のユーザー、グループ、カスタムロールを管理することを可能にします。SCIM グループは W&B Teams にマップされます。

SCIM API は `<host-url>/scim/` でアクセス可能であり、[RC7643 プロトコル](https://www.rfc-editor.org/rfc/rfc7643) に記載されているフィールドのサブセットを持つ `/Users` と `/Groups` エンドポイントをサポートします。さらに、公式の SCIM スキーマには含まれていない `/Roles` エンドポイントもあります。W&B はカスタムロールの自動管理をサポートするために `/Roles` エンドポイントを追加しました。

:::info
SCIM API は [Dedicated Cloud](../hosting-options/dedicated_cloud.md)、[Self-managed instances](../hosting-options/self-managed.md)、および [SaaS Cloud](../hosting-options/saas_cloud.md) を含むすべてのホスティングオプションに適用されます。SaaS Cloud では、組織管理者が SCIM API リクエストが正しい組織に送信されるように、ユーザー設定でデフォルトの組織を設定する必要があります。この設定は、ユーザー設定内の `SCIM API Organization` セクションにあります。
:::

## 認証

SCIM API はインスタンスまたは組織の管理者が API キーを使用してベーシック認証でアクセスできます。ベーシック認証では、HTTP リクエストに `Authorization` ヘッダーを含め、その後にスペースと `username:password` のベース64エンコード文字列を含めます。ここで `password` は API キーです。例えば、`demo:p@55w0rd` として認証するには、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` となります。

## ユーザーリソース

SCIM ユーザーリソースは W&B ユーザーにマップされます。

### ユーザーを取得

- **Endpoint:** **`<host-url>/scim/Users/{id}`**
- **Method**: GET
- **Description**: ユーザーの固有 ID を指定することで、[SaaS Cloud](../hosting-options/saas_cloud.md) 組織や [Dedicated Cloud](../hosting-options/dedicated_cloud.md)、[Self-managed](../hosting-options/self-managed.md) インスタンスの特定のユーザー情報を取得します。
- **Request Example**:

```bash
GET /scim/Users/abc
```

- **Response Example**:

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

### ユーザー一覧を取得

- **Endpoint:** **`<host-url>/scim/Users`**
- **Method**: GET
- **Description**: [SaaS Cloud](../hosting-options/saas_cloud.md) 組織や [Dedicated Cloud](../hosting-options/dedicated_cloud.md)、 [Self-managed](../hosting-options/self-managed.md) インスタンス内の全ユーザーリストを取得します。
- **Request Example**:

```bash
GET /scim/Users
```

- **Response Example**:

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

### ユーザーを作成

- **Endpoint**: **`<host-url>/scim/Users`**
- **Method**: POST
- **Description**: 新しいユーザーリソースを作成します。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| emails | Multi-Valued Array | Yes (Make sure `primary` email is set) |
| userName | String | Yes |
- **Request Example**:

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

- **Response Example**:

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

### ユーザーを削除

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: DELETE
- **Description**: ユーザーの固有 ID を指定して、[SaaS Cloud](../hosting-options/saas_cloud.md) 組織や [Dedicated Cloud](../hosting-options/dedicated_cloud.md)、 [Self-managed](../hosting-options/self-managed.md) インスタンスから完全に削除します。必要に応じて、再度ユーザーを追加するには [Create user](#create-user) API を使用します。
- **Request Example**:

:::note
ユーザーを一時的に無効化するには、`PATCH` エンドポイントを使用する [Deactivate user](#deactivate-user) API を参照してください。
:::

```bash
DELETE /scim/Users/abc
```

- **Response Example**:

```json
(Status 204)
```

### ユーザーを無効化

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: PATCH
- **Description**: ユーザーの固有 ID を指定して、[Dedicated Cloud](../hosting-options/dedicated_cloud.md) または [Self-managed](../hosting-options/self-managed.md) インスタンス内のユーザーを一時的に無効化します。必要に応じて、ユーザーを再活性化するには [Reactivate user](#reactivate-user) API を使用します。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| op | String | Type of operation. The only allowed value is `replace`. |
| value | Object | Object `{"active": false}` indicating that the user should be deactivated. |

:::note
ユーザーの無効化および再活性化操作は [SaaS Cloud](../hosting-options/saas_cloud.md) ではサポートされていません。
:::

- **Request Example**:

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

- **Response Example**:
This returns the User object.

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

### ユーザーを再活性化

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: PATCH
- **Description**: ユーザーの固有 ID を指定して、[Dedicated Cloud](../hosting-options/dedicated_cloud.md) または [Self-managed](../hosting-options/self-managed.md) インスタンス内のユーザーを再活性化します。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| op | String | Type of operation. The only allowed value is `replace`. |
| value | Object | Object `{"active": true}` indicating that the user should be reactivated. |

:::note
ユーザーの無効化および再活性化操作は [SaaS Cloud](../hosting-options/saas_cloud.md) ではサポートされていません。
:::

- **Request Example**:

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

- **Response Example**:
This returns the User object.

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

### 組織レベルのロールをユーザーに割り当て

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: PATCH
- **Description**: 組織レベルのロールをユーザーに割り当てます。そのロールは `admin`、`viewer` もしくは `member` のいずれかです。[ここ](./manage-users#invite-users) に記載されています。[SaaS Cloud](../hosting-options/saas_cloud.md) の場合、SCIM API の正しい組織がユーザー設定で設定されていることを確認してください。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| op | String | Type of operation. The only allowed value is `replace`. |
| path | String | The scope at which role assignment operation takes effect. The only allowed value is `organizationRole`. |
| value | String | The predefined organization-level role to assign to the user. It can be one of `admin`, `viewer` or `member`. This field is case insensitive for predefined roles. |
- **Request Example**:

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

- **Response Example**:
This returns the User object.

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

### チームレベルのロールをユーザーに割り当て

- **Endpoint**: **`<host-url>/scim/Users/{id}`**
- **Method**: PATCH
- **Description**: チームレベルのロールをユーザーに割り当てます。そのロールは `admin`、`viewer`、`member` もしくはカスタムロールです。[ここ](./manage-users#team-roles)に記載されています。[SaaS Cloud](../hosting-options/saas_cloud.md) の場合、SCIM API の正しい組織がユーザー設定で設定されていることを確認してください。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| op | String | Type of operation. The only allowed value is `replace`. |
| path | String | The scope at which role assignment operation takes effect. The only allowed value is `teamRoles`. |
| value | Object array | A one-object array where the object consists of `teamName` and `roleName` attributes. The `teamName` is the name of the team where the user holds the role, and `roleName` can be one of `admin`, `viewer`, `member` or a custom role. This field is case insensitive for predefined roles and case sensitive for custom roles. |
- **Request Example**:

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

- **Response Example**:
This returns the User object.

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

## グループリソース

SCIM グループリソースは W&B Teams にマップされます。つまり、W&B デプロイメントで SCIM グループを作成すると、W&B Team が作成されます。他のグループエンドポイントにも同様のことが適用されます。

### チームを取得

- **Endpoint**: **`<host-url>/scim/Groups/{id}`**
- **Method**: GET
- **Description**: チームの固有 ID を指定してチーム情報を取得します。
- **Request Example**:

```bash
GET /scim/Groups/ghi
```

- **Response Example**:

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

### チーム一覧を取得

- **Endpoint**: **`<host-url>/scim/Groups`**
- **Method**: GET
- **Description**: チームの一覧を取得します。
- **Request Example**:

```bash
GET /scim/Groups
```

- **Response Example**:

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

### チームを作成

- **Endpoint**: **`<host-url>/scim/Groups`**
- **Method**: POST
- **Description**: 新しいチームリソースを作成します。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| displayName | String | Yes |
| members | Multi-Valued Array | Yes (`value` sub-field is required and maps to a user ID) |
- **Request Example**:

`dev-user2` をメンバーとする `wandb-support` というチームを作成する例。

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

- **Response Example**:

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

### チームを更新

- **Endpoint**: **`<host-url>/scim/Groups/{id}`**
- **Method**: PATCH
- **Description**: 既存のチームのメンバーシップリストを更新します。
- **Supported Operations**: `add` member, `remove` member
- **Request Example**:

`dev-user2` を `wandb-devs` に追加する例

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

- **Response Example**:

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

### チームを削除

- チームには関連するデータがあるため、現在 SCIM API によるチーム削除はサポートされていません。すべてを削除するか確認するためにアプリからチームを削除してください。

## ロールリソース

SCIM ロールリソースは W&B カスタムロールにマップされます。前述のとおり、`/Roles` エンドポイントは公式の SCIM スキーマには含まれていませんが、W&B 組織内のカスタムロールの自動管理をサポートするために追加されました。

### カスタムロールを取得

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: GET
- **Description**: ロールの固有 ID を指定してカスタムロールの情報を取得します。
- **Request Example**:

```bash
GET /scim/Roles/abc
```

- **Response Example**:

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

### カスタムロール一覧を取得

- **Endpoint**: **`<host-url>/scim/Roles`**
- **Method**: GET
- **Description**: W&B 組織におけるすべてのカスタムロールの情報を取得します。
- **Request Example**:

```bash
GET /scim/Roles
```

- **Response Example**:

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

### カスタムロールを作成

- **Endpoint**: **`<host-url>/scim/Roles`**
- **Method**: POST
- **Description**: W&B 組織内に新しいカスタムロールを作成します。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| name | String | Name of the custom role |
| description | String | Description of the custom role |
| permissions | Object array | Array of permission objects where each object includes a `name` string field that has value of the form `w&bobject:operation`. For example, a permission object for delete operation on W&B runs would have `name` as `run:delete`. |
| inheritedFrom | String | The predefined role which the custom role would inherit from. It can either be `member` or `viewer`. |
- **Request Example**:

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

- **Response Example**:

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

### カスタムロールを削除

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: DELETE
- **Description**: W&B 組織内のカスタムロールを削除します。この操作は慎重に使用してください。カスタムロールの継承元の定義済みロールは、操作前にカスタムロールが割り当てられていた全ユーザーに再割り当てされます。
- **Request Example**:

```bash
DELETE /scim/Roles/abc
```

- **Response Example**:

```bash
(Status 204)
```

### カスタムロールのパーミッションを更新

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: PATCH
- **Description**: W&B 組織内でカスタムロールのカスタムパーミッションを追加または削除します。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| operations | Object array | Array of operation objects |
| op | String | Type of operation within the operation object. It can either be `add` or `remove`. |
| path | String | Static field in the operation object. Only value allowed is `permissions`. |
| value | Object array | Array of permission objects where each object includes a `name` string field that has value of the form `w&bobject:operation`. For example, a permission object for delete operation on W&B runs would have `name` as `run:delete`. |
- **Request Example**:

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

- **Response Example**:

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

### カスタムロールのメタデータを更新

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: PUT
- **Description**: W&B 組織内のカスタムロールの名前、説明、もしくは継承元のロールを更新します。この操作は、既存のカスタムパーミッションには影響しません。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| name | String | Name of the custom role |
| description | String | Description of the custom role |
| inheritedFrom | String | The predefined role which the custom role inherits from. It can either be `member` or `viewer`. |
- **Request Example**:

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

- **Response Example**:

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
