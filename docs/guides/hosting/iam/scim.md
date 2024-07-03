---
displayed_sidebar: default
---

# SCIM

System for Cross-domain Identity Management (SCIM) APIを使用すると、インスタンスまたは組織の管理者はW&B組織内のユーザー、グループ、およびカスタムロールを管理できます。SCIMグループはW&Bのチームにマップされます。

SCIM APIは `<host-url>/scim/` でアクセス可能であり、[RC7643プロトコル](https://www.rfc-editor.org/rfc/rfc7643)にあるフィールドのサブセットをサポートする`/Users` と `/Groups` エンドポイントを提供します。さらに、公式のSCIMスキーマには含まれない `/Roles` エンドポイントも含まれています。W&Bは、W&B組織内でのカスタムロールの自動管理をサポートするために `/Roles` エンドポイントを追加しています。

:::info
SCIM APIは、[Dedicated Cloud](../hosting-options/dedicated_cloud.md)、[Self-managed instances](../hosting-options/self-managed.md)、および[SaaS Cloud](../hosting-options/saas_cloud.md)を含むすべてのホスティングオプションに適用されます。SaaS Cloudでは、組織管理者がSCIM APIリクエストが正しい組織に向かうことを確認するために、ユーザー設定でデフォルトの組織を設定する必要があります。この設定はユーザー設定の`SCIM API Organization`セクションで利用可能です。
:::

## 認証

SCIM APIはインスタンスまたは組織の管理者が自分のAPIキーを使用して基本認証でアクセス可能です。基本認証を使用すると、HTTPリクエストを`Authorization`ヘッダーと共に送信し、ヘッダーには`Basic`という単語のあとにスペースと`username:password`をbase64エンコードした文字列が続きます。ここで`password`はAPIキーです。例えば、`demo:p@55w0rd`として認証するには、ヘッダーは`Authorization: Basic ZGVtbzpwQDU1dzByZA==` となります。

## Userリソース

SCIMユーザーリソースはW&Bのユーザーにマップされます。

### ユーザー情報の取得

- **エンドポイント:** **`<host-url>/scim/Users/{id}`**
- **メソッド**: GET
- **説明**: [SaaS Cloud](../hosting-options/saas_cloud.md)組織またはあなたの[Dedicated Cloud](../hosting-options/dedicated_cloud.md)または[Self-managed](../hosting-options/self-managed.md)インスタンスの特定のユーザーの情報をユーザーのユニークIDで取得します。
- **リクエスト例**:

```bash
GET /scim/Users/abc
```

- **レスポンス例**:

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

### ユーザーリストの取得

- **エンドポイント:** **`<host-url>/scim/Users`**
- **メソッド**: GET
- **説明**: [SaaS Cloud](../hosting-options/saas_cloud.md)組織またはあなたの[Dedicated Cloud](../hosting-options/dedicated_cloud.md)または[Self-managed](../hosting-options/self-managed.md)インスタンスのすべてのユーザーリストを取得します。
- **リクエスト例**:

```bash
GET /scim/Users
```

- **レスポンス例**:

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

### ユーザーの作成

- **エンドポイント**: **`<host-url>/scim/Users`**
- **メソッド**: POST
- **説明**: 新しいユーザーリソースの作成。
- **対応フィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| emails | Multi-Valued Array | はい (プライマリメールはセットされるべきです) |
| userName | String | はい |
- **リクエスト例**:

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

- **レスポンス例**:

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

### ユーザーの削除

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: DELETE
- **説明**: [SaaS Cloud](../hosting-options/saas_cloud.md)組織や[Dedicated Cloud](../hosting-options/dedicated_cloud.md)または[Self-managed](../hosting-options/self-managed.md)インスタンスからユーザーを完全に削除します。必要に応じて、[Create user](#create-user) APIを使用してユーザーを再度組織またはインスタンスに追加します。
- **リクエスト例**:

:::note
一時的にユーザーを無効にするには、`PATCH`エンドポイントを使用する[Deactivate user](#deactivate-user) APIを参照してください。
:::

```bash
DELETE /scim/Users/abc
```

- **レスポンス例**:

```json
(Status 204)
```

### ユーザーの無効化

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: [Dedicated Cloud](../hosting-options/dedicated_cloud.md)または[Self-managed](../hosting-options/self-managed.md)インスタンスのユーザーを一時的に無効にします(ユーザーのユニークIDを提供)。必要に応じて[Reactivate user](#reactivate-user) APIを使用してユーザーを再有効化します。
- **対応フィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | String | 操作タイプ。許可される唯一の値は `replace` です。 |
| value | Object | ユーザーを無効にすべきことを示すオブジェクト`{"active": false}`。 |

:::note
ユーザーの無効化と再有効化の操作は[SaaS Cloud](../hosting-options/saas_cloud.md)ではサポートされていません。
:::

- **リクエスト例**:

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

- **レスポンス例**:
このレスポンスはUserオブジェクトを返します。

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

### ユーザーの再有効化

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: 無効化されたユーザーを再度有効にします (Dedicated Cloud](../hosting-options/dedicated_cloud.md) または [Self-managed](../hosting-options/self-managed.md)インスタンスの中の)。ユーザーのユニークIDを提供。
- **対応フィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | String | 操作タイプ。許可される唯一の値は `replace` です。 |
| value | Object | ユーザーを再度有効にすべきことを示すオブジェクト `{"active": true}`。 |

:::note
ユーザーの無効化と再有効化の操作は[SaaS Cloud](../hosting-options/saas_cloud.md)ではサポートされていません。
:::

- **リクエスト例**:

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

- **レスポンス例**:
このレスポンスはUserオブジェクトを返します。

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

### ユーザーに組織レベルのロールを割り当てる

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: ユーザーに組織レベルのロールを割り当てます。ロールは `admin`, `viewer` または`member`のいずれかです。[こちら](./manage-users#invite-users)で説明されています。 [SaaS Cloud](../hosting-options/saas_cloud.md)の場合、SCIM API用に正しい組織をユーザー設定で設定していることを確認してください。
- **対応フィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | String | 操作タイプ。許可される唯一の値は `replace` です。 |
| path | String | ロール割り当て操作が実施されるスコープ。許可される唯一の値は `organizationRole` です。 |
| value | String | ユーザーに割り当てられる事前定義の組織レベルのロール。 `admin`, `viewer`, `member`のいずれか。このフィールドは事前定義のロールに対しては大文字小文字を区別しません。 |
- **リクエスト例**:

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
            "value": "admin" // ユーザーの組織スコープロールをadminに設定します
        }
    ]
}
```

- **レスポンス例**:
このレスポンスはUserオブジェクトを返します。

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
    "teamRoles": [  // ユーザーが所属するすべてのチームでのロールを返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 組織スコープでのユーザーのロールを返します
}
```

### ユーザーにチームレベルのロールを割り当てる

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: ユーザーにチームレベルのロールを割り当てます。ロールは `admin`, `viewer`, `member`または[こちら](./manage-users#team-roles)で説明されているカスタムロールのいずれかです。 [SaaS Cloud](../hosting-options/saas_cloud.md)の場合、SCIM APIの正しい組織をユーザー設定で設定していることを確認してください。
- **対応フィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | String | 操作タイプ。許可される唯一の値は `replace` です。 |
| path | String | ロール割り当て操作が実施されるスコープ。許可される唯一の値は `teamRoles`です。 |
| value | オブジェクト配列 | 1つのオブジェクト配列の中で、オブジェクトは`teamName`と`roleName`属性で構成されます。`teamName`はユーザーが役割を持つチームの名前であり、`roleName`は`admin`, `viewer`, `member`またはカスタムロールのいずれかです。このフィールドは事前定義のロールに対しては大文字小文字を区別しませんが、カスタムロールに対しては区別します。 |
- **リクエスト例**:

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
                    "roleName": "admin", // 事前定義ロールの場合は大文字小文字を区別しませんがカスタムロールの場合は区別します
                    "teamName": "team1" // チームteam1でユーザーのロールをadminに設定します
                }
            ]
        }
    ]
}
```

- **レスポンス例**:
このレスポンスはUserオブジェクトを返します。

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
    "teamRoles": [  // ユーザーが所属するすべてのチームでのロールを返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 組織スコープでのユーザーのロールを返します
}
```

## Groupリソース

SCIMグループリソースはW&Bのチームにマップされます。つまり、SCIMグループをW&Bのデプロイメントで作成すると、W&Bのチームが作成されます。他のグループエンドポイントも同様です。

### チーム情報の取得

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: GET
- **説明**: チームのユニークIDを提供してチーム情報を取得します。
- **リクエスト例**:

```bash
GET /scim/Groups/ghi
```

- **レスポンス例**:

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

### チームリストの取得

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: GET
- **説明**: チームリストを取得します。
- **リクエスト例**:

```bash
GET /scim/Groups
```

- **レスポンス例**:

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

### チームの作成

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: POST
- **説明**: 新しいチームリソースを作成します。
- **対応フィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| displayName | String | はい |
| members | Multi-Valued Array | はい (`value` サブフィールドは必須で、ユーザーIDにマップされます) |
- **リクエスト例**:

`wandb-support`というチームを`dev-user2`をメンバーとして作成する。

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

- **レスポンス例**:

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

### チームの更新

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: PATCH
- **説明**: 既存のチームのメンバーシップリストを更新します。
- **対応操作**: `add` メンバー, `remove` メンバー
- **リクエスト例**:

`wandb-devs`に`dev-user2`を追加する

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

- **レスポンス例**:

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

### チームの削除

- チームの削除はSCIM APIでは現在サポートされていません。チームには追加のデータがリンクされているためです。すべてのデータが削除されることを確認するには、アプリからチームを削除してください。

## Roleリソース

SCIMロールリソースはW&Bカスタムロールにマップされます。前述のように、`/Roles`エンドポイントは公式SCIMスキーマには含まれておらず、W&BはW&B組織内でカスタムロールの自動管理をサポートするために`/Roles`エンドポイントを追加しています。

### カスタムロールの取得

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: GET
- **説明**: ロールのユニークIDを提供してカスタムロールの情報を取得します。
- **リクエスト例**:

```bash
GET /scim/Roles/abc
```

- **レスポンス例**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 事前定義ロールを示しています。
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
            "isInherited": true // member事前定義ロールから継承されています
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 管理者によって追加されたカスタム権限
        }
    ],
    "schemas": [
        ""
    ]
}
```

### カスタムロールの一覧

- **エンドポイント**: **`<host-url>/scim/Roles`**
- **メソッド**: GET
- **説明**: W&B組織内のすべてのカスタムロールの情報を取得します
- **リクエスト例**:

```bash
GET /scim/Roles
```

- **レスポンス例**:

```bash
(Status 200)
```

```json
{
   "Resources": [
        {
            "description": "A sample custom role for example",
            "id": "Um9sZTo3",
            "inheritedFrom": "member", // カスタムロールが継承する事前定義されたロールを示します。
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
                    "isInherited": true // member事前定義ロールから継承されています
                },
                ...
                ...
                {
                    "name": "project:update",
                    "isInherited": false // 管理者によって追加されたカスタム権限
                }
            ],
            "schemas": [
                ""
            ]
        },
        {
            "description": "Another sample custom role for example",
            "id": "Um9sZToxMg==",
            "inheritedFrom": "viewer", // カスタムロールが継承する事前定義されたロールを示します。
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
                    "isInherited": true // viewer事前定義ロールから継承されています
                },
                ...
                ...
                {
                    "name": "run:stop",
                    "isInherited": false // 管理者によって追加されたカスタム権限
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

### カスタムロールの作成

- **エンドポイント**: **`<host-url>/scim/Roles`**
- **メソッド**: POST
- **説明**: W&B組織内に新しいカスタムロールを作成します。
- **対応フィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| name | String | カスタムロールの名前 |
| description | String | カスタムロールの説明 |
| permissions | オブジェクト配列 | 各オブジェクトが `name` という文字列フィールドを含む権限オブジェクトの配列。この値は `w&bobject:operation` の形式となります。例えば、W&B runsの削除操作に対して `name` を `run:delete` とします。 |
| inheritedFrom | String | カスタムロールが継承する事前定義されたロール。`member` または `viewer` のいずれかです。 |
- **リクエスト例**:

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

- **レスポンス例**:

```bash
(Status 201)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 事前定義ロールを示しています。
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
            "isInherited": true // member事前定義ロールから継承されています。
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 管理者によって追加されたカスタム権限。
        }
    ],
    "schemas": [
        ""
    ]
}
```

### カスタムロールの削除

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: DELETE
- **説明**: W&B組織内のカスタムロールを削除します。 **慎重に使用してください**。カスタムロールから継承した事前定義ロールは、操作前にカスタムロールが割り当てられていたすべてのユーザーに再度割り当てられます。
- **リクエスト例**:

```bash
DELETE /scim/Roles/abc
```

- **レスポンス例**:

```bash
(Status 204)
```

### カスタムロール権限の更新

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: PATCH
- **説明**: W&B組織内のカスタムロールでカスタム権限を追加または削除します。
- **対応フィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| operations | オブジェクト配列 | 操作オブジェクトの配列 |
| op | String | 操作オブジェクト内の操作タイプ。`add` または `remove` のどちらか。 |
| path | String | 操作オブジェクト内の静的フィールド。許可される唯一の値は `permissions` です。 |
| value | オブジェクト配列 | 各オブジェクトが `name` という文字列フィールドを含む権限オブジェクトの配列。この値は `w&bobject:operation` の形式となります。例えば、W&B runsの削除操作に対して `name` を `run:delete` とします。 |
- **リクエスト例**:

```bash
PATCH /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add", // 操作タイプを示しています。もう一つの可能な値は `remove` です。
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

- **レスポンス例**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 事前定義ロールを示しています。
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
            "isInherited": true // member事前定義ロールから継承されています。
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 更新前に管理者によって追加された既存のカスタム権限。
        },
        {
            "name": "project:delete",
            "isInherited": false // 更新により管理者によって追加された新しいカスタム権限。
        }
    ],
    "schemas": [
        ""
    ]
}
```

### カスタムロールのメタデータを更新

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: PUT
- **説明**: W&B組織内のカスタムロールの名前、説明、または継承元のロールを更新します。この操作は、既存の非継承カスタム権限には影響しません。
- **対応フィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| name | String | カスタムロールの名前 |
| description | String | カスタムロールの説明 |
| inheritedFrom | String | カスタムロールが継承する事前定義ロール。`member` または `viewer` のいずれか。 |
- **リクエスト例**:

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

- **レスポンス例**:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example but now based on viewer", // 要求に応じて説明が変更されました。
    "id": "Um9sZTo3",
    "inheritedFrom": "viewer", // 要求に従って変更された事前定義ロールを示します。
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
            "isInherited": true // viewer事前定義ロールから継承されています。
        },
        ... // member事前定義ロールにあるがviewerにはない権限は更新後には継承されません
        {
            "name": "project:update",
            "isInherited": false // 管理者によって追加されたカスタム権限
        },
        {
            "name": "project:delete",
            "isInherited": false // 管理者によって追加されたカスタム権限
        }
    ],
    "schemas": [
        ""
    ]
}
```