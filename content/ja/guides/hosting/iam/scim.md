---
title: Manage users, groups, and roles with SCIM
menu:
  default:
    identifier: ja-guides-hosting-iam-scim
    parent: identity-and-access-management-iam
weight: 4
---

System for Cross-domain Identity Management (SCIM) API を使用すると、インスタンスまたは Organization の管理者は、W&B Organization 内の ユーザー 、 グループ 、およびカスタムロールを管理できます。SCIM グループ は W&B Teams にマッピングされます。

SCIM API は `<host-url>/scim/` でアクセスでき、[RC7643 プロトコル](https://www.rfc-editor.org/rfc/rfc7643) にあるフィールドのサブセットを持つ `/Users` および `/Groups` エンドポイントをサポートします。さらに、公式の SCIM スキーマには含まれていない `/Roles` エンドポイントも含まれています。W&B は、W&B Organization におけるカスタムロールの自動管理をサポートするために `/Roles` エンドポイントを追加します。

{{% alert %}}
SCIM API は、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}), [Self-managed instances]({{< relref path="/guides/hosting/hosting-options/self_managed.md" lang="ja" >}}), および [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) を含むすべてのホスティングオプションに適用されます。SaaS Cloud では、Organization 管理者は、SCIM API リクエストが適切な Organization に送信されるように、 ユーザー 設定でデフォルトの Organization を構成する必要があります。この設定は、 ユーザー 設定内の `SCIM API Organization` セクションにあります。
{{% /alert %}}

## 認証

Organization またはインスタンス管理者は、 APIキー を使用して基本認証で SCIM API に アクセス できます。HTTP リクエストの `Authorization` ヘッダーを文字列 `Basic` の後にスペース、そして `username:API-KEY` の形式で base-64 エンコードされた文字列に設定します。言い換えれば、 ユーザー 名と APIキー を `:` 文字で区切られた ユーザー の 値 に置き換え、その 結果 を base-64 でエンコードします。たとえば、`demo:p@55w0rd` として認証するには、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` にする必要があります。

## ユーザー リソース

SCIM ユーザー リソース は W&B ユーザー にマッピングされます。

### ユーザー の取得

- **エンドポイント:** **`<host-url>/scim/Users/{id}`**
- **メソッド**: GET
- **説明**: ユーザー の一意の ID を指定して、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) Organization 、 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) 、または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self_managed.md" lang="ja" >}}) インスタンス内の特定の ユーザー の情報を取得します。
- **リクエスト の例**:

```bash
GET /scim/Users/abc
```

- **レスポンス の例**:

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

### ユーザー の一覧表示

- **エンドポイント:** **`<host-url>/scim/Users`**
- **メソッド**: GET
- **説明**: [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) Organization 、 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) 、または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self_managed.md" lang="ja" >}}) インスタンス内のすべての ユーザー の一覧を取得します。
- **リクエスト の例**:

```bash
GET /scim/Users
```

- **レスポンス の例**:

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

### ユーザー の作成

- **エンドポイント**: **`<host-url>/scim/Users`**
- **メソッド**: POST
- **説明**: 新しい ユーザー リソース を作成します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| emails | 複数値の配列 | はい ( `primary` メールが設定されていることを確認してください) |
| userName | 文字列 | はい |
- **リクエスト の例**:

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

- **レスポンス の例**:

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

### ユーザー の削除

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: DELETE
- **説明**: ユーザー の一意の ID を指定して、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) Organization 、 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) 、または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self_managed.md" lang="ja" >}}) インスタンスから ユーザー を完全に削除します。必要に応じて、[ユーザー の作成]({{< relref path="#create-user" lang="ja" >}}) API を使用して、 ユーザー を Organization またはインスタンスに再度追加します。
- **リクエスト の例**:

{{% alert %}}
ユーザー を一時的に非アクティブ化するには、`PATCH` エンドポイントを使用する [ユーザー の非アクティブ化]({{< relref path="#deactivate-user" lang="ja" >}}) API を参照してください。
{{% /alert %}}

```bash
DELETE /scim/Users/abc
```

- **レスポンス の例**:

```json
(Status 204)
```

### ユーザー の非アクティブ化

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: ユーザー の一意の ID を指定して、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self_managed.md" lang="ja" >}}) インスタンス内の ユーザー を一時的に非アクティブ化します。必要に応じて、[ユーザー の再アクティブ化]({{< relref path="#reactivate-user" lang="ja" >}}) API を使用して、 ユーザー を再アクティブ化します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作のタイプ。許可される 値 は `replace` のみです。 |
| value | オブジェクト | ユーザー を非アクティブ化する必要があることを示すオブジェクト `{"active": false}` 。 |

{{% alert %}}
ユーザー の非アクティブ化および再アクティブ化操作は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではサポートされていません。
{{% /alert %}}

- **リクエスト の例**:

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

- **レスポンス の例**:
これは ユーザー オブジェクト を返します。

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

### ユーザー の再アクティブ化

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: ユーザー の一意の ID を指定して、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self_managed.md" lang="ja" >}}) インスタンス内の非アクティブ化された ユーザー を再アクティブ化します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作のタイプ。許可される 値 は `replace` のみです。 |
| value | オブジェクト | ユーザー を再アクティブ化する必要があることを示すオブジェクト `{"active": true}` 。 |

{{% alert %}}
ユーザー の非アクティブ化および再アクティブ化操作は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではサポートされていません。
{{% /alert %}}

- **リクエスト の例**:

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

- **レスポンス の例**:
これは ユーザー オブジェクト を返します。

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

### ユーザー に Organization レベルのロールを割り当てる

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: ユーザー に Organization レベルのロールを割り当てます。ロールは、[こちら]({{< relref path="access-management/manage-organization.md#invite-a-user" lang="ja" >}}) で説明されているように、`admin`、`viewer`、または `member` のいずれかになります。[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、 ユーザー 設定で SCIM API 用に正しい Organization が構成されていることを確認してください。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作のタイプ。許可される 値 は `replace` のみです。 |
| path | 文字列 | ロール割り当て操作が有効になる スコープ 。許可される 値 は `organizationRole` のみです。 |
| value | 文字列 | ユーザー に割り当てる事前定義された Organization レベルのロール。`admin`、`viewer`、または `member` のいずれかになります。このフィールドでは、事前定義されたロールの大文字と小文字は区別されません。 |
- **リクエスト の例**:

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

- **レスポンス の例**:
これは ユーザー オブジェクト を返します。

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

### ユーザー に Team レベルのロールを割り当てる

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: ユーザー に Team レベルのロールを割り当てます。ロールは、[こちら]({{< relref path="access-management/manage-organization.md#assign-or-update-a-team-members-role" lang="ja" >}}) で説明されているように、`admin`、`viewer`、`member`、またはカスタムロールのいずれかになります。[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、 ユーザー 設定で SCIM API 用に正しい Organization が構成されていることを確認してください。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作のタイプ。許可される 値 は `replace` のみです。 |
| path | 文字列 | ロール割り当て操作が有効になる スコープ 。許可される 値 は `teamRoles` のみです。 |
| value | オブジェクト配列 | オブジェクトが `teamName` および `roleName` 属性で構成される 1 つのオブジェクト配列。`teamName` は ユーザー がロールを保持する Team の名前であり、`roleName` は `admin`、`viewer`、`member`、またはカスタムロールのいずれかになります。このフィールドでは、事前定義されたロールの大文字と小文字は区別されず、カスタムロールの大文字と小文字は区別されます。 |
- **リクエスト の例**:

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

- **レスポンス の例**:
これは ユーザー オブジェクト を返します。

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

## グループ リソース

SCIM グループ リソース は W&B Teams にマッピングされます。つまり、W&B デプロイメント で SCIM グループ を作成すると、W&B Team が作成されます。他の グループ エンドポイント にも同じことが当てはまります。

### Team の取得

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: GET
- **説明**: Team の一意の ID を指定して、Team 情報を取得します。
- **リクエスト の例**:

```bash
GET /scim/Groups/ghi
```

- **レスポンス の例**:

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

### Team の一覧表示

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: GET
- **説明**: Team の一覧を取得します。
- **リクエスト の例**:

```bash
GET /scim/Groups
```

- **レスポンス の例**:

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

### Team の作成

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: POST
- **説明**: 新しい Team リソース を作成します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| displayName | 文字列 | はい |
| members | 複数値の配列 | はい ( `value` サブフィールドは必須で、 ユーザー ID にマッピングされます) |
- **リクエスト の例**:

`wandb-support` という名前の Team を `dev-user2` をメンバーとして作成します。

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

- **レスポンス の例**:

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

### Team の更新

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: PATCH
- **説明**: 既存の Team のメンバーシップ リスト を更新します。
- **サポートされている操作**: メンバー の `add` 、メンバー の `remove`
- **リクエスト の例**:

`dev-user2` を `wandb-devs` に追加します。

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

- **レスポンス の例**:

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

### Team の削除

- Team の削除は、Team にリンクされている追加の データ があるため、現在 SCIM API ではサポートされていません。すべてを削除することを確認するには、アプリケーションから Team を削除してください。

## ロール リソース

SCIM ロール リソース は W&B カスタムロール にマッピングされます。前述のように、`/Roles` エンドポイント は公式の SCIM スキーマ の一部ではありません。W&B は、W&B Organization におけるカスタムロールの自動管理をサポートするために `/Roles` エンドポイントを追加します。

### カスタムロール の取得

- **エンドポイント:** **`<host-url>/scim/Roles/{id}`**
- **メソッド**: GET
- **説明**: ロールの一意の ID を指定して、カスタムロール の情報を取得します。
- **リクエスト の例**:

```bash
GET /scim/Roles/abc
```

- **レスポンス の例**:

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

### カスタムロール の一覧表示

- **エンドポイント:** **`<host-url>/scim/Roles`**
- **メソッド**: GET
- **説明**: W&B Organization 内のすべてのカスタムロール の情報を取得します。
- **リクエスト の例**:

```bash
GET /scim/Roles
```

- **レスポンス の例**:

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

### カスタムロール の作成

- **エンドポイント**: **`<host-url>/scim/Roles`**
- **メソッド**: POST
- **説明**: W&B Organization に新しいカスタムロール を作成します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| name | 文字列 | カスタムロール の名前 |
| description | 文字列 | カスタムロール の説明 |
| permissions | オブジェクト配列 | 各オブジェクトに `w&bobject:operation` の形式の値を持つ `name` 文字列フィールドが含まれる、 権限 オブジェクト の配列。たとえば、W&B Runs での削除操作に対する 権限 オブジェクト の `name` は `run:delete` になります。 |
| inheritedFrom | 文字列 | カスタムロール が継承する事前定義されたロール。`member` または `viewer` のいずれかになります。 |
- **リクエスト の例**:

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

- **レスポンス の例**:

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

### カスタムロール の削除

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: DELETE
- **説明**: W&B Organization でカスタムロール を削除します。**注意して使用してください**。カスタムロール が継承した事前定義されたロールは、操作前にカスタムロール が割り当てられていたすべての ユーザー に割り当てられるようになりました。
- **リクエスト の例**:

```bash
DELETE /scim/Roles/abc
```

- **レスポンス の例**:

```bash
(Status 204)
```

### カスタムロール の 権限 を更新する

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: PATCH
- **説明**: W&B Organization のカスタムロール でカスタム 権限 を追加または削除します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| operations | オブジェクト配列 | 操作オブジェクト の配列 |
| op | 文字列 | 操作オブジェクト 内の操作のタイプ。`add` または `remove` のいずれかになります。 |
| path | 文字列 | 操作オブジェクト 内の静的フィールド。許可される 値 は `permissions` のみです。 |
| value | オブジェクト配列 | 各オブジェクトに `w&bobject:operation` の形式の値を持つ `name` 文字列フィールドが含まれる、 権限 オブジェクト の配列。たとえば、W&B Runs での削除操作に対する 権限 オブジェクト の `name` は `run:delete` になります。 |
- **リクエスト の例**:

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

- **レスポンス の例**:

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

### カスタムロール の メタデータ を更新する

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: PUT
- **説明**: W&B Organization のカスタムロール の名前、説明、または継承されたロールを更新します。この操作は、カスタムロール 内の既存の、つまり継承されていないカスタム 権限 には影響しません。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| name | 文字列 | カスタムロール の名前 |
| description | 文字列 | カスタムロール の説明 |
| inheritedFrom | 文字列 | カスタムロール が継承する事前定義されたロール。`member` または `viewer` のいずれかになります。 |
- **リクエスト の例**:

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

- **レスポンス の例**:

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