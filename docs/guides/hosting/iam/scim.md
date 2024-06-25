---
displayed_sidebar: default
---


# SCIM

SCIM (System for Cross-domain Identity Management) APIは、インスタンスまたは組織の管理者が自分のW&B組織内のユーザー、グループ、およびカスタムロールを管理することを可能にします。SCIMグループはW&B Teamsにマップされます。

SCIM APIは `<host-url>/scim/` でアクセス可能で、[RC7643プロトコル](https://www.rfc-editor.org/rfc/rfc7643)に含まれるフィールドのサブセットを持つ `/Users` および `/Groups` エンドポイントをサポートしています。さらに、公式のSCIMスキーマには含まれていない `/Roles` エンドポイントも含まれており、W&B組織内のカスタムロールの自動管理をサポートしています。

:::info
SCIM APIは、専用クラウド、セルフマネージドデプロイメント、マルチテナントクラウドを含むすべてのホスティングオプションに適用されます。マルチテナントクラウドでは、組織管理者はユーザー設定でデフォルトの組織を設定して、SCIM APIリクエストが正しい組織に送信されるようにする必要があります。設定は `SCIM API Organization` セクションで利用可能です。
:::

## 認証

SCIM APIは、インスタンスまたは組織の管理者が自分のAPIキーを使ってベーシック認証を使用することでアクセス可能です。ベーシック認証を使用する場合、HTTPリクエストの `Authorization` ヘッダーに `Basic` という単語の後にスペースを置き、`username:password` の形式の文字列をbase64エンコードしたものを設定します。`password` はAPIキーです。例えば、`demo:p@55w0rd` として認証する場合、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` となります。

## ユーザーリソース

SCIMユーザーリソースはW&B Usersにマップされます。

### ユーザー取得

- **エンドポイント:** **`<host-url>/scim/Users/{id}`**
- **メソッド**: GET
- **説明**: ユニークなIDを提供してユーザー情報を取得します。
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

### ユーザーリスト取得

- **エンドポイント:** **`<host-url>/scim/Users`**
- **メソッド**: GET
- **説明**: ユーザーのリストを取得します。
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

### ユーザー作成

- **エンドポイント**: **`<host-url>/scim/Users`**
- **メソッド**: POST
- **説明**: 新しいユーザーリソースを作成します。
- **サポートされているフィールド**:

| フィールド | 型 | 必須 |
| --- | --- | --- |
| emails | マルチ値配列 | はい（`primary` メールがセットされていることを確認してください） |
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

### ユーザーの非アクティブ化

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: DELETE
- **説明**: ユニークなIDを提供してユーザーを非アクティブ化します。
- **リクエスト例**:

```bash
DELETE /scim/Users/abc
```

- **レスポンス例**:

```json
(Status 204)
```

### ユーザーの再アクティブ化

- 以前に非アクティブ化されたユーザーの再アクティブ化は、現在のSCIM APIではサポートされていません。

### 組織レベルのロールをユーザーに割り当て

- **エンドポイント**: **`<host-url>/scim/Users/{userId}`**
- **メソッド**: PATCH
- **説明**: 組織レベルのロールをユーザーに割り当てます。ロールは `admin`、`viewer` または `member` のいずれかです。詳しくは[こちら](./manage-users#invite-users)を参照してください。Public Cloudの場合、SCIM APIの正しい組織がユーザー設定で構成されていることを確認してください。
- **サポートされているフィールド**:

| フィールド | 型 | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作の種類。唯一許された値は `replace` です。 |
| path | 文字列 | ロール割り当て操作が行われるスコープ。唯一許された値は `organizationRole` です。 |
| value | 文字列 | 割り当てる組織レベルのロールを定義します。`admin`、`viewer`、または `member` のいずれかです。このフィールドは事前定義されたロールの場合は大文字小文字を区別しません。 |
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
            "value": "admin" // ユーザーの組織スコープのロールをadminに設定します
        }
    ]
}
```

- **レスポンス例**:
User オブジェクトを返します。

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
    "teamRoles": [  // ユーザーが所属するすべてのTeamsにおけるユーザーのロールを返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 組織スコープにおけるユーザーのロールを返します
}
```

### チームレベルのロールをユーザーに割り当て

- **エンドポイント**: **`<host-url>/scim/Users/{userId}`**
- **メソッド**: PATCH
- **説明**: チームレベルのロールをユーザーに割り当てます。ロールは `admin`、`viewer`、`member` または [こちら](./manage-users#team-roles)で説明されているカスタムロールのいずれかです。Public Cloudの場合、SCIM APIの正しい組織がユーザー設定で構成されていることを確認してください。
- **サポートされているフィールド**:

| フィールド | 型 | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作の種類。唯一許された値は `replace` です。 |
| path | 文字列 | ロール割り当て操作が行われるスコープ。唯一許された値は `teamRoles` です。 |
| value | オブジェクト配列 | オブジェクトが `teamName` および `roleName` 属性を含む1つのオブジェクトの配列です。`teamName` はユーザーがロールを持つチーム名で、`roleName` は `admin`、`viewer`、`member` またはカスタムロールのいずれかです。このフィールドは事前定義されたロールの場合は大文字小文字を区別せず、カスタムロールの場合は区別します。 |
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
                    "roleName": "admin", // 事前定義されたロールの場合は大文字小文字を区別せず、カスタムロールの場合は区別します
                    "teamName": "team1" // ユーザーのチームteam1におけるロールをadminに設定します
                }
            ]
        }
    ]
}
```

- **レスポンス例**:
User オブジェクトを返します。

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
    "teamRoles": [  // ユーザーが所属するすべてのTeamsにおけるユーザーのロールを返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 組織スコープにおけるユーザーのロールを返します
}
```

## グループリソース

SCIMグループリソースはW&B Teamsにマップされます。つまり、W&B デプロイメントでSCIMグループを作成すると、W&B Teamが作成されます。他のグループエンドポイントにも同じことが適用されます。

### チーム取得

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: GET
- **説明**: ユニークなIDを提供してチーム情報を取得します。
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

### チームリスト取得

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: GET
- **説明**: チームのリストを取得します。
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

### チーム作成

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: POST
- **説明**: 新しいチームリソースを作成します。
- **サポートされているフィールド**:

| フィールド | 型 | 必須 |
| --- | --- | --- |
| displayName | 文字列 | はい |
| members | マルチ値配列 | はい (`value` サブフィールドが必須でユーザーIDにマップされます) |
- **リクエスト例**:

`wandb-support` という名前のチームを `dev-user2` として作成します。

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

### チーム更新

- **エンドポイント**: **`<host-url