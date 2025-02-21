---
title: Manage users, groups, and roles with SCIM
menu:
  default:
    identifier: ja-guides-hosting-iam-scim
    parent: identity-and-access-management-iam
weight: 4
---


システム間ドメインアイデンティティ管理（SCIM）APIは、インスタンスまたは組織の管理者が W&B 組織内で ユーザー、グループ、およびカスタムロールを管理するのを可能にします。SCIMグループは W&B チームにマップされます。

SCIM API は `<host-url>/scim/` でアクセス可能で、`/Users` および `/Groups` エンドポイントをサポートし、[RC7643プ​​ロトコル](https://www.rfc-editor.org/rfc/rfc7643) にあるフィールドのサブセットを持っています。さらに、公式の SCIM スキーマにはない `/Roles` エンドポイントを含んでいます。 W&B は、W&B 組織内のカスタムロールの自動化された管理をサポートするために、`/Roles` エンドポイントを追加しています。

{{% alert %}}
SCIM API は、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、[セルフマネージドインスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})、および [SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) などのすべてのホスティングオプションに適用されます。 SaaS クラウドでは、組織の管理者はユーザー 設定でデフォルトの組織を設定し、SCIM API リクエストが正しい組織に送信されるようにする必要があります。この設定はユーザー 設定内の `SCIM API 組織` セクションで利用可能です。
{{% /alert %}}

## 認証

組織またはインスタンスの管理者は、APIキーを使用して基本認証を使用し、SCIM API にアクセスできます。HTTP リクエストの `Authorization` ヘッダーを文字列 `Basic` で設定し、その後にスペースを入れ、`username:API-KEY` 形式の文字列を base-64 エンコードします。つまり、ユーザー名と APIキーを `:` 文字で区切って置き換え、その結果を base-64 エンコードします。たとえば、`demo:p@55w0rd` として認証するには、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` となります。

## ユーザーリソース

SCIM ユーザーリソースは W&B ユーザーにマップされます。

### ユーザー情報の取得

- **エンドポイント:** **`<host-url>/scim/Users/{id}`**
- **メソッド**: GET
- **説明**: [SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})の組織や,[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})または[セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内で、ユーザーの一意のIDを提供して特定のユーザーの情報を取得します。
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
- **説明**: [SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})の組織や、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、または[セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内のすべてのユーザーのリストを取得します。
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
- **説明**: 新しいユーザーリソースを作成します。
- **サポートされているフィールド**:

| Field | Type | 必須 |
| --- | --- | --- |
| emails | マルチバリュー配列 | 必須 (必ず`primary` のメールが設定されていることを確認) |
| userName | 文字列 | 必須 |
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
- **説明**: [SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})の組織や、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、または[セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内のユーザーをユーザーの一意の ID を提供して完全に削除します。必要に応じて、再び組織やインスタンスにユーザーを追加するには [ユーザーの作成]({{< relref path="#create-user" lang="ja" >}}) API を使用してください。
- **リクエスト例**:

{{% alert %}}
ユーザーを一時的に無効化するには、`PATCH` エンドポイントを使用する [ユーザーの無効化]({{< relref path="#deactivate-user" lang="ja" >}}) API を参照してください。
{{% /alert %}}

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
- **説明**: [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})や[セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})インスタンス内のユーザーをユーザーの一意の ID を提供して一時的に無効化します。必要に応じて、[ユーザーの再有効化]({{< relref path="#reactivate-user" lang="ja" >}}) API を使用してユーザーを再有効化します。
- **サポートされているフィールド**:

| Field | Type | 必須 |
| --- | --- | --- |
| op | 文字列 | オペレーションの種類。許可される唯一の値は `replace` です。 |
| value | オブジェクト | ユーザーを無効化することを示すオブジェクト `{"active": false}` です。 |

{{% alert %}}
ユーザーの無効化および再有効化の操作は [SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではサポートされていません。
{{% /alert %}}

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
これはユーザーオブジェクトを返します。

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
- **説明**: [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})や[セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})インスタンス内で無効化されたユーザーをユーザーの一意の ID を提供して再有効化します。
- **サポートされているフィールド**:

| Field | Type | 必須 |
| --- | --- | --- |
| op | 文字列 | オペレーションの種類。許可される唯一の値は `replace` です。 |
| value | オブジェクト | ユーザーを再有効化することを示すオブジェクト `{"active": true}` です。 |

{{% alert %}}
ユーザーの無効化および再有効化の操作は [SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではサポートされていません。
{{% /alert %}}

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
これはユーザーオブジェクトを返します。

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

### 組織レベルのロールをユーザーに割り当てる

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: ユーザーに組織レベルのロールを割り当てます。ロールは、[こちら]({{< relref path="access-management/manage-organization.md#invite-a-user" lang="ja" >}})で説明されているように、`admin`、`viewer`、`member` のいずれかです。 [SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})の場合、SCIM API の正しい組織がユーザー 設定で設定されていることを確認してください。
- **サポートされているフィールド**:

| Field | Type | 必須 |
| --- | --- | --- |
| op | 文字列 | オペレーションの種類。許可される唯一の値は `replace` です。 |
| path | 文字列 | ロール割り当てオペレーションの影響を与えるスコープ。許可される唯一の値は `organizationRole` です。 |
| value | 文字列 | ユーザーに割り当てる事前定義の組織レベルのロール。`admin`、`viewer`、または `member` のいずれかの値になります。このフィールドは事前定義のロールに対しては大文字小文字を区別しません。 |
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
            "value": "admin" // ユーザーの組織スコープのロールを admin に設定します
        }
    ]
}
```

- **レスポンス例**:
これはユーザーオブジェクトを返します。

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
    "teamRoles": [  // ユーザーが所属するすべてのチームに対するロールを返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 組織スコープでのユーザーのロールを返します
}
```

### チームレベルのロールをユーザーに割り当てる

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: ユーザーにチームレベルのロールを割り当てます。ロールは、[こちら]({{< relref path="access-management/manage-organization.md#assign-or-update-a-team-members-role" lang="ja" >}})で説明されているように、`admin`、`viewer`、`member` またはカスタムロールのいずれかです。[SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})の場合、SCIM API の正しい組織がユーザー 設定で設定されていることを確認してください。
- **サポートされているフィールド**:

| Field | Type | 必須 |
| --- | --- | --- |
| op | 文字列 | オペレーションの種類。許可される唯一の値は `replace` です。 |
| path | 文字列 | ロール割り当てオペレーションの影響を与えるスコープ。許可される唯一の値は `teamRoles` です。 |
| value | オブジェクト配列 | `teamName` および `roleName` 属性を含むオブジェクトの1オブジェクト配列で、`teamName`はユーザーがロールを持つチームの名前で、`roleName`は`admin`、`viewer`、`member`またはカスタムロールのいずれかである。事前定義のロールに対しては大文字小文字を区別せず、カスタムロールに対しては区別します。 |
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
                    "roleName": "admin", // 事前定義のロールに対しては大文字小文字を区別せず、カスタムロールに対しては区別します
                    "teamName": "team1" // チーム team1 におけるユーザーのロールを admin に設定します
                }
            ]
        }
    ]
}
```

- **レスポンス例**:
これはユーザーオブジェクトを返します。

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
    "teamRoles": [  // ユーザーが所属するすべてのチームに対するロールを返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // 組織スコープでのユーザーのロールを返します
}
```

## グループリソース

SCIM グループリソースは W&B チームにマップされます。つまり、W&B デプロイメントで SCIM グループを作成すると、W&B チームが作成されます。他のグループエンドポイントにも同様に適用されます。

### チーム情報の取得

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: GET
- **説明**: チーム情報をチームの一意の ID を提供して取得します。
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

### チームの作成

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: POST
- **説明**: 新しいチームリソースを作成します。
- **サポートされているフィールド**:

| Field | Type | 必須 |
| --- | --- | --- |
| displayName | 文字列 | 必須 |
| members | マルチバリュー配列 | 必須 (`value` サブフィールドは必須で、ユーザー ID にマッピングされます) |
- **リクエスト例**:

`dev-user2` をメンバーとする `wandb-support` というチームを作成

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
- **サポートされている操作**: メンバーを `追加`、メンバーを `削除`
- **リクエスト例**:

`wandb-devs` に `dev-user2` を追加

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

- 現在、SCIM API ではチームの削除はサポートされていません。チームにリンクされた追加のデータが存在するためです。すべて削除することを確認するにはアプリからチームを削除してください。

## ロールリソース

SCIM ロールリソースは W&B のカスタムロールにマップされます。前述のように、`/Roles` エンドポイントは公式の SCIM スキーマの一部ではなく、W&B は W&B 組織内のカスタムロールの自動管理をサポートするために `/Roles` エンドポイントを追加しています。

### カスタムロールの取得

- **エンドポイント:** **`<host-url>/scim/Roles/{id}`**
- **メソッド**: GET
- **説明**: ロールの一意の ID を提供してカスタムロールの情報を取得します。
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
    "inheritedFrom": "member", // 事前定義のロールを示します
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
            "isInherited": true // member 事前定義のロールから継承
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 管理者によって追加されたカスタムパーミッション
        }
    ],
    "schemas": [
        ""
    ]
}
```

### カスタムロールのリスト取得

- **エンドポイント:** **`<host-url>/scim/Roles`**
- **メソッド**: GET
- **説明**: W&B 組織内のすべてのカスタムロールの情報を取得します
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
            "inheritedFrom": "member", // カスタムロールが継承する事前定義のロールを示します
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
                    "isInherited": true // member 事前定義のロールから継承
                },
                ...
                ...
                {
                    "name": "project:update",
                    "isInherited": false // 管理者によって追加されたカスタムパーミッション
                }
            ],
            "schemas": [
                ""
            ]
        },
        {
            "description": "Another sample custom role for example",
            "id": "Um9sZToxMg==",
            "inheritedFrom": "viewer", // カスタムロールが継承する事前定義のロールを示します
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
                    "isInherited": true // viewer 事前定義のロールから継承
                },
                ...
                ...
                {
                    "name": "run:stop",
                    "isInherited": false // 管理者によって追加されたカスタムパーミッション
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
- **説明**: W&B 組織内に新しいカスタムロールを作成します。
- **サポートされているフィールド**:

| Field | Type | 必須 |
| --- | --- | --- |
| name | 文字列 | カスタムロールの名前 |
| description | 文字列 | カスタムロールの説明 |
| permissions | オブジェクト配列 | 各オブジェクトに `name` 文字列フィールドを含むパーミッションオブジェクトの配列で、`w&bobject:operation` の形式の値を持ちます。たとえば、W&B の runs に対する削除操作のパーミッションオブジェクトは `run:delete` を `name` として持ちます。 |
| inheritedFrom | 文字列 | カスタムロールが継承する事前定義のロール。それは `member` か `viewer` のいずれかです。 |
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
    "inheritedFrom": "member", // 事前定義の役割を示します
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
            "isInherited": true // member 事前定義の役割から継承
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

### カスタムロールの削除

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: DELETE
- **説明**: W&B 組織内のカスタムロールを削除します。 **注意して使用してください**。カスタムロールが継承していた事前定義のロールが、操作の前にカスタムロールが割り当てられていたすべてのユーザーに割り当てられます。
- **リクエスト例**:

```bash
DELETE /scim/Roles/abc
```

- **レスポンス例**:

```bash
(Status 204)
```

### カスタムロールの権限を更新

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: PATCH
- **説明**: W&B 組織内のカスタムロールにカスタム権限を追加または削除します。
- **サポートされているフィールド**:

| Field | Type | 必須 |
| --- | --- | --- |
| operations | オブジェクト配列 | 操作オブジェクトの配列 |
| op | 文字列 | 操作オブジェクト内の操作の種類。`add` または `remove` のどちらかです。 |
| path | 文字列 | 操作オブジェクト内の固定フィールド。許可される値は `permissions` のみです。 |
| value | オブジェクト配列 | 各オブジェクトに `name` 文字列フィールドを含む権限オブジェクトの配列で、`w&bobject:operation` の形式の値を持ちます。たとえば、W&B の runs に対する削除操作の権限オブジェクトは `run:delete` を `name` として持ちます。 |
- **リクエスト例**:

```bash
PATCH /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add", // 操作の種類を示します。他の可能な値は `remove` です
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
    "inheritedFrom": "member", // 事前定義の役割を示します
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
            "isInherited": true // member 事前定義の役割から継承
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 更新前に管理者によって追加された既存のカスタム権限
        },
        {
            "name": "project:delete",
            "isInherited": false // 更新の一環として管理者によって追加された新しいカスタム権限
        }
    ],
    "schemas": [
        ""
    ]
}
```

### カスタムロールメタデータの更新

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: PUT
- **説明**: W&B 組織内のカスタムロールの名前、説明、または継承されたロールを更新します。この操作は、カスタムロール内の既存の、つまり継承されないカスタム権限には影響を与えません。
- **サポートされているフィールド**:

| Field | Type | 必須 |
| --- | --- | --- |
| name | 文字列 | カスタムロールの名前 |
| description | 文字列 | カスタムロールの説明 |
| inheritedFrom | 文字列 | カスタムロールが継承する事前定義のロール。それは `member` か `viewer` のいずれかです。 |
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
    "description": "A sample custom role for example but now based on viewer", // リクエストに従って説明を変更
    "id": "Um9sZTo3",
    "inheritedFrom": "viewer", // リクエストに従って変更された事前定義の役割を示します
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
            "isInherited": true // viewer 事前定義の役割から継承
        },
        ... // 更新後、member の事前定義ロールに含まれるが、viewer には含まれない権限は継承されません
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