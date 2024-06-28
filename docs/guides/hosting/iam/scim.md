---
displayed_sidebar: default
---


# SCIM

System for Cross-domain Identity Management (SCIM) API は、インスタンスまたは組織の管理者が W&B 組織内のユーザー、グループ、およびカスタム役割を管理するためのものです。SCIM グループは W&B チームにマッピングされます。

SCIM API は `<host-url>/scim/` でアクセスでき、[RC7643 プロトコル](https://www.rfc-editor.org/rfc/rfc7643) に含まれるフィールドのサブセットを持つ `/Users` および `/Groups` エンドポイントをサポートします。また、公式の SCIM スキーマには含まれていない `/Roles` エンドポイントも含まれています。W&B は、W&B 組織内のカスタム役割の自動管理をサポートするために `/Roles` エンドポイントを追加しています。

:::info
SCIM API は 専用クラウド、自己管理デプロイメント、マルチテナントクラウドを含むすべてのホスティングオプションに適用されます。マルチテナントクラウドでは、組織管理者はユーザー設定でデフォルトの組織を設定して、SCIM API リクエストが正しい組織に送信されるようにする必要があります。この設定は `SCIM API Organization` セクションで利用できます。
:::

## 認証

SCIM API は、APIキーを使用した基本認証を使用して、インスタンスまたは組織の管理者によってアクセスできます。基本認証では、HTTP リクエストに `Authorization` ヘッダーを含め、その後にスペースとベース64エンコードされた `username:password` 文字列を含めます。ここで `password` は APIキーです。例えば、`demo:p@55w0rd` を認証するには、ヘッダーが `Authorization: Basic ZGVtbzpwQDU1dzByZA==` である必要があります。

## ユーザーリソース

SCIM ユーザーリソースは W&B ユーザーにマッピングされます。

### ユーザー情報の取得

- **エンドポイント:** **`<host-url>/scim/Users/{id}`**
- **メソッド**: GET
- **説明**: ユーザーの一意の ID を提供してユーザー情報を取得します。
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
- **説明**: ユーザーリストを取得します。
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
- **説明**: 新しいユーザーリソースを作成します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| emails | 多価配列 | はい（`primary` メールが設定されていることを確認してください） |
| userName | 文字列 | はい |
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

### ユーザーの無効化

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: DELETE
- **説明**: ユーザーの一意の ID を提供してユーザーを無効化します。
- **リクエスト例**:

```bash
DELETE /scim/Users/abc
```

- **レスポンス例**:

```json
(Status 204)
```

### ユーザーの再活性化

- 以前に無効化されたユーザーの再活性化は現在 SCIM API ではサポートされていません。

### 組織レベルの役割をユーザーに割り当て

- **エンドポイント**: **`<host-url>/scim/Users/{userId}`**
- **メソッド**: PATCH
- **説明**: 組織レベルの役割をユーザーに割り当てます。役割は、[こちら](./manage-users#invite-users) で説明されている `admin`、`viewer`、または `member` のいずれかです。Public Cloud の場合、SCIM API に対して正しい組織がユーザー設定で構成されていることを確認してください。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作の種類。唯一許可される値は `replace` です。 |
| path | 文字列 | 役割割り当て操作が影響を受けるスコープ。唯一許可される値は `organizationRole` です。 |
| value | 文字列 | ユーザーに割り当てる事前定義された組織レベルの役割。`admin`、`viewer`、または `member` のいずれかになります。このフィールドは事前定義された役割については大文字小文字を区別せず、カスタム役割については大文字小文字を区別します。 |
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
            "value": "admin" // ユーザーの組織スコープ内の役割を管理者に設定します
        }
    ]
}
```

- **レスポンス例**:
ユーザーオブジェクトを返します。

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
    "teamRoles": [  // ユーザーが所属するすべてのチームにおけるユーザーの役割を返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 組織スコープでのユーザーの役割を返します
}
```

### チームレベルの役割をユーザーに割り当て

- **エンドポイント**: **`<host-url>/scim/Users/{userId}`**
- **メソッド**: PATCH
- **説明**: チームレベルの役割をユーザーに割り当てます。役割は、[こちら](./manage-users#team-roles) で説明されている `admin`、`viewer`、`member` またはカスタム役割のいずれかです。Public Cloud の場合、SCIM API に対して正しい組織がユーザー設定で構成されていることを確認してください。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作の種類。唯一許可される値は `replace` です。 |
| path | 文字列 | 役割割り当て操作が影響を与えるスコープ。唯一許可される値は `teamRoles` です。 |
| value | オブジェクト配列 | オブジェクトが `teamName` と `roleName` 属性から成る一つの配列オブジェクト。`teamName` はユーザーが役割を持つチームの名前、`roleName` は `admin`、`viewer`、`member`またはカスタム役割のいずれかです。このフィールドは事前定義された役割については大文字小文字を区別せず、カスタム役割については大文字小文字を区別します。 |
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
                    "roleName": "admin", // 事前定義された役割の名前は大文字小文字を区別せず、カスタム役割の名前は区別します
                    "teamName": "team1" // ユーザーのチーム team1 内での役割を管理者に設定します
                }
            ]
        }
    ]
}
```

- **レスポンス例**:
ユーザーオブジェクトを返します。

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
    "teamRoles": [  // ユーザーが所属するすべてのチームにおけるユーザーの役割を返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 組織スコープでのユーザーの役割を返します
}
```

## グループリソース

SCIM グループリソースは W&B チームにマッピングされます。つまり、W&B デプロイメントで SCIM グループを作成すると、W&B チームが作成されます。他のグループエンドポイントにも同じことが適用されます。

### チーム情報の取得

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: GET
- **説明**: チームの一意の ID を提供してチーム情報を取得します。
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
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| displayName | 文字列 | はい |
| members | 多価配列 | はい (`value` サブフィールドは必須であり、ユーザー ID にマッピングされます) |
- **リクエスト例**:

`wandb-support` というチームを `dev-user2` をメンバーとして作成します。

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


### Update team

- **Endpoint**: **`<host-url>/scim/Groups/{id}`**
- **Method**: PATCH
- **Description**: 既存のチームのメンバーシップリストを更新します。
- **Supported Operations**: メンバーを `add`、メンバーを `remove`
- **Request Example**:

`dev-user2` を `wandb-devs` に追加

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

### Delete team

- SCIM APIではチームの削除は現在サポートされていません。チームにリンクされた追加データが存在するためです。すべてを削除することを確認するには、アプリからチームを削除してください。

## Role resource

SCIMロールリソースはW&Bのカスタムロールにマッピングされます。先述のとおり、`/Roles` エンドポイントは公式のSCIMスキーマの一部ではなく、W&BはW&B組織におけるカスタムロールの自動管理をサポートするために `/Roles` エンドポイントを追加しています。

### Get custom role

- **Endpoint:** **`<host-url>/scim/Roles/{id}`**
- **Method**: GET
- **Description**: ロールの一意のIDを提供してカスタムロールの情報を取得します。
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
    "inheritedFrom": "member", // 事前定義されたロール
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
            "isInherited": true // メンバーから継承された定義済みロール
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

### List custom roles

- **Endpoint:** **`<host-url>/scim/Roles`**
- **Method**: GET
- **Description**: W&B組織内のすべてのカスタムロールの情報を取得します。
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
            "inheritedFrom": "member", // カスタムロールが継承する事前定義されたロール
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
                    "isInherited": true // メンバーから継承された定義済みロール
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
            "inheritedFrom": "viewer", // カスタムロールが継承する事前定義されたロール
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
                    "isInherited": true // viewerから継承された定義済みロール
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

### Create custom role

- **Endpoint**: **`<host-url>/scim/Roles`**
- **Method**: POST
- **Description**: W&B組織内に新しいカスタムロールを作成します。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| name | String | カスタムロールの名前 |
| description | String | カスタムロールの説明 |
| permissions | Object array | 各オブジェクトに`name`文字列フィールドを含む権限オブジェクトの配列。このフィールドの値は`w&bobject:operation`の形式になります。例えば、W&B runs での削除操作の権限オブジェクトは`name`が`run:delete`となります。 |
| inheritedFrom | String | カスタムロールが継承する事前定義されたロール。`member`または`viewer`のいずれかです。 |
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
    "inheritedFrom": "member", // 事前定義されたロール
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
            "isInherited": true // メンバーから継承された定義済みロール
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

### Delete custom role

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: DELETE
- **Description**: W&B組織内のカスタムロールを削除します。 **慎重に使用してください**。カスタムロールが継承していた事前定義されたロールは、操作前にカスタムロールに割り当てられていたすべてのユーザーに再割り当てされます。
- **Request Example**:

```bash
DELETE /scim/Roles/abc
```

- **Response Example**:

```bash
(Status 204)
```

### Update custom role permissions

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: PATCH
- **Description**: W&B組織内のカスタムロールにカスタム権限を追加または削除します。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| operations | Object array | 操作オブジェクトの配列 |
| op | String | 操作オブジェクト内の操作の種類。`add`または`remove`のいずれかです。 |
| path | String | 操作オブジェクト内の静的フィールド。許可される値は`permissions`のみです。 |
| value | Object array | 各オブジェクトに`name`文字列フィールドを含む権限オブジェクトの配列。このフィールドの値は`w&bobject:operation`の形式になります。例えば、W&B runs での削除操作の権限オブジェクトは`name`が`run:delete`となります。 |
- **Request Example**:

```bash
PATCH /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add", // 操作の種類を示します。その他の可能な値は`remove`
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
    "inheritedFrom": "member", // 事前定義されたロール
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
            "isInherited": true // メンバーから継承された定義済みロール
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 更新前に管理者によって追加された既存のカスタム権限
        },
        {
            "name": "project:delete",
            "isInherited": false // 更新の一環として追加された新しいカスタム権限
        }
    ],
    "schemas": [
        ""
    ]
}
```

### Update custom role metadata

- **Endpoint**: **`<host-url>/scim/Roles/{id}`**
- **Method**: PUT
- **Description**: W&B組織内のカスタムロールの名前、説明、または継承されたロールを更新します。この操作は、既存のカスタム権限には影響を与えません。
- **Supported Fields**:

| Field | Type | Required |
| --- | --- | --- |
| name | String | カスタムロールの名前 |
| description | String | カスタムロールの説明 |
| inheritedFrom | String | カスタムロールが継承する事前定義されたロール。`member`または`viewer`のいずれかです。 |
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
    "description": "A sample custom role for example but now based on viewer", // リクエストに基づいて説明を変更
    "id": "Um9sZTo3",
    "inheritedFrom": "viewer", // リクエストに基づいて変更された事前定義されたロール
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
            "isInherited": true // viewerから継承された定義済みロール
        },
        ... // メンバーの事前定義されたロールに含まれるが、更新後にはviewerに含まれない権限は継承されません
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