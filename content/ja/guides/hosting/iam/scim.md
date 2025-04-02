---
title: Manage users, groups, and roles with SCIM
menu:
  default:
    identifier: ja-guides-hosting-iam-scim
    parent: identity-and-access-management-iam
weight: 4
---

{{% alert %}}
[SCIM の動作をデモする動画](https://www.youtube.com/watch?v=Nw3QBqV0I-o)（12 分）をご覧ください。
{{% /alert %}}

System for Cross-domain Identity Management（SCIM）API を使用すると、インスタンスまたは organization の管理者は、W&B organization 内の user、グループ、およびカスタムロールを管理できます。SCIM グループは W&B の Teams にマッピングされます。

SCIM API は `<host-url>/scim/` でアクセスでき、[RC7643 プロトコル](https://www.rfc-editor.org/rfc/rfc7643)にあるフィールドのサブセットを使用して、`/Users` および `/Groups` エンドポイントをサポートします。さらに、公式の SCIM スキーマにはない `/Roles` エンドポイントが含まれています。W&B は、W&B organization でのカスタムロールの自動管理をサポートするために、`/Roles` エンドポイントを追加します。

{{% alert %}}
複数のエンタープライズ [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) organization の管理者である場合は、SCIM API リクエストの送信先となる organization を構成する必要があります。プロフィール画像をクリックし、**User Settings** をクリックします。設定の名前は **Default API organization** です。これは、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}), [Self-managed instances]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}), および [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) を含む、すべてのホスティングオプションで必要です。SaaS Cloud では、organization 管理者は、SCIM API リクエストが正しい organization に送信されるように、user 設定でデフォルトの organization を構成する必要があります。

選択したホスティングオプションによって、このページの例で使用される `<host-url>` プレースホルダーの value が決定されます。

さらに、例では user ID（`abc` や `def` など）を使用します。実際のリクエストとレスポンスでは、user ID にハッシュ value が設定されます。
{{% /alert %}}

## 認証

organization またはインスタンスの管理者は、 APIキー を使用して基本認証で SCIM API にアクセスできます。HTTP リクエストの `Authorization` ヘッダーを文字列 `Basic` に設定し、その後にスペース、次に `username:API-KEY` の形式で base-64 エンコードされた文字列を設定します。つまり、username と APIキー を `:` 文字で区切られた value に置き換え、その結果を base-64 エンコードします。たとえば、`demo:p@55w0rd` として認証するには、ヘッダーを `Authorization: Basic ZGVtbzpwQDU1dzByZA==` にする必要があります。

## User リソース

SCIM の user リソースは W&B の Users にマッピングされます。

### User を取得

- **エンドポイント:** **`<host-url>/scim/Users/{id}`**
- **メソッド**: GET
- **説明**: user の一意の ID を指定して、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) organization または [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) あるいは [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内の特定の user の情報を取得します。
- **リクエストの例**:

```bash
GET /scim/Users/abc
```

- **レスポンスの例**:

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

### Users をリスト表示

- **エンドポイント:** **`<host-url>/scim/Users`**
- **メソッド**: GET
- **説明**: [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) organization または [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) あるいは [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内のすべての Users のリストを取得します。
- **リクエストの例**:

```bash
GET /scim/Users
```

- **レスポンスの例**:

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

### User を作成

- **エンドポイント**: **`<host-url>/scim/Users`**
- **メソッド**: POST
- **説明**: 新しい user リソースを作成します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| emails | 複数値の配列 | はい（`primary` メールが設定されていることを確認してください） |
| userName | 文字列 | はい |
- **リクエストの例**:

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

- **レスポンスの例**:

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

### User を削除

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: DELETE
- **説明**: user の一意の ID を指定して、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) organization または [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) あるいは [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスから user を完全に削除します。必要に応じて、[User を作成]({{< relref path="#create-user" lang="ja" >}}) API を使用して、user を organization またはインスタンスに再度追加します。
- **リクエストの例**:

{{% alert %}}
User を一時的に非アクティブ化するには、`PATCH` エンドポイントを使用する [User を非アクティブ化]({{< relref path="#deactivate-user" lang="ja" >}}) API を参照してください。
{{% /alert %}}

```bash
DELETE /scim/Users/abc
```

- **レスポンスの例**:

```json
(Status 204)
```

### User を非アクティブ化

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: user の一意の ID を指定して、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内の user を一時的に非アクティブ化します。必要に応じて、[User を再アクティブ化]({{< relref path="#reactivate-user" lang="ja" >}}) API を使用して、user を再アクティブ化します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作のタイプ。許可される value は `replace` のみです。 |
| value | オブジェクト | User を非アクティブ化することを示すオブジェクト `{"active": false}`。 |

{{% alert %}}
User の非アクティブ化および再アクティブ化の操作は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではサポートされていません。
{{% /alert %}}

- **リクエストの例**:

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

- **レスポンスの例**:
これにより、User オブジェクトが返されます。

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

### User を再アクティブ化

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: user の一意の ID を指定して、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内の非アクティブ化された user を再アクティブ化します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作のタイプ。許可される value は `replace` のみです。 |
| value | オブジェクト | User を再アクティブ化することを示すオブジェクト `{"active": true}`。 |

{{% alert %}}
User の非アクティブ化および再アクティブ化の操作は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではサポートされていません。
{{% /alert %}}

- **リクエストの例**:

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

- **レスポンスの例**:
これにより、User オブジェクトが返されます。

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

### User に organization レベルのロールを割り当てる

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: user に organization レベルのロールを割り当てます。ロールは、[こちら]({{< relref path="access-management/manage-organization.md#invite-a-user" lang="ja" >}})で説明されているように、`admin`、`viewer`、または `member` のいずれかになります。[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合は、user 設定で SCIM API 用に正しい organization を構成していることを確認してください。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作のタイプ。許可される value は `replace` のみです。 |
| path | 文字列 | ロールの割り当て操作が有効になるスコープ。許可される value は `organizationRole` のみです。 |
| value | 文字列 | user に割り当てる定義済みの organization レベルのロール。`admin`、`viewer`、または `member` のいずれかになります。このフィールドでは、定義済みのロールの大文字と小文字は区別されません。 |
- **リクエストの例**:

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
            "value": "admin" // user の organization スコープのロールを admin に設定します
        }
    ]
}
```

- **レスポンスの例**:
これにより、User オブジェクトが返されます。

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
    "teamRoles": [  // user が所属するすべての Teams における user のロールを返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // organization スコープにおける user のロールを返します
}
```

### User に Team レベルのロールを割り当てる

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: user に Team レベルのロールを割り当てます。ロールは、[こちら]({{< relref path="access-management/manage-organization.md#assign-or-update-a-team-members-role" lang="ja" >}})で説明されているように、`admin`、`viewer`、`member`、またはカスタムロールのいずれかになります。[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合は、user 設定で SCIM API 用に正しい organization を構成していることを確認してください。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| op | 文字列 | 操作のタイプ。許可される value は `replace` のみです。 |
| path | 文字列 | ロールの割り当て操作が有効になるスコープ。許可される value は `teamRoles` のみです。 |
| value | オブジェクト配列 | オブジェクトが `teamName` および `roleName` 属性で構成される 1 オブジェクト配列。`teamName` は user がロールを保持する Team の名前で、`roleName` は `admin`、`viewer`、`member`、またはカスタムロールのいずれかになります。このフィールドでは、定義済みのロールの大文字と小文字は区別されず、カスタムロールの大文字と小文字は区別されます。 |
- **リクエストの例**:

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
                    "roleName": "admin", // 定義済みのロールの場合、ロール名の大文字と小文字は区別されず、カスタムロールの場合は大文字と小文字が区別されます
                    "teamName": "team1" // Team team1 における user のロールを admin に設定します
                }
            ]
        }
    ]
}
```

- **レスポンスの例**:
これにより、User オブジェクトが返されます。

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
    "teamRoles": [  // user が所属するすべての Teams における user のロールを返します
        {
            "teamName": "team1",
            "roleName": "admin"
        }
    ],
    "organizationRole": "admin" // organization スコープにおける user のロールを返します
}
```

## Group リソース

SCIM のグループリソースは W&B の Teams にマッピングされます。つまり、W&B デプロイメントで SCIM グループを作成すると、W&B の Team が作成されます。他のグループエンドポイントにも同じことが当てはまります。

### Team を取得

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: GET
- **説明**: Team の一意の ID を指定して、Team 情報を取得します。
- **リクエストの例**:

```bash
GET /scim/Groups/ghi
```

- **レスポンスの例**:

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

### Teams をリスト表示

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: GET
- **説明**: Teams のリストを取得します。
- **リクエストの例**:

```bash
GET /scim/Groups
```

- **レスポンスの例**:

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

### Team を作成

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: POST
- **説明**: 新しい Team リソースを作成します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| displayName | 文字列 | はい |
| members | 複数値の配列 | はい（`value` サブフィールドは必須で、user ID にマッピングされます） |
- **リクエストの例**:

`dev-user2` をメンバーとして、`wandb-support` という Team を作成します。

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

- **レスポンスの例**:

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

### Team を更新

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: PATCH
- **説明**: 既存の Team のメンバーシップリストを更新します。
- **サポートされている操作**: メンバーの `add`、メンバーの `remove`
- **リクエストの例**:

`dev-user2` を `wandb-devs` に追加します

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

- **レスポンスの例**:

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

### Team を削除

- Team の削除は、Teams にリンクされている追加データがあるため、現在 SCIM API ではサポートされていません。すべてを削除することを確認するには、アプリから Teams を削除します。

## Role リソース

SCIM のロールリソースは W&B のカスタムロールにマッピングされます。前述のように、`/Roles` エンドポイントは公式の SCIM スキーマの一部ではありません。W&B は、W&B organization でのカスタムロールの自動管理をサポートするために、`/Roles` エンドポイントを追加します。

### カスタムロールを取得

- **エンドポイント:** **`<host-url>/scim/Roles/{id}`**
- **メソッド**: GET
- **説明**: ロールの一意の ID を指定して、カスタムロールの情報を取得します。
- **リクエストの例**:

```bash
GET /scim/Roles/abc
```

- **レスポンスの例**:

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

### カスタムロールをリスト表示

- **エンドポイント:** **`<host-url>/scim/Roles`**
- **メソッド**: GET
- **説明**: W&B organization 内のすべてのカスタムロールの情報を取得します
- **リクエストの例**:

```bash
GET /scim/Roles
```

- **レスポンスの例**:

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

- **エンドポイント**: **`<host-url>/scim/Roles`**
- **メソッド**: POST
- **説明**: W&B organization に新しいカスタムロールを作成します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| name | 文字列 | カスタムロールの名前 |
| description | 文字列 | カスタムロールの説明 |
| permissions | オブジェクト配列 | 各オブジェクトに `w&bobject:operation` の形式の value を持つ `name` 文字列フィールドが含まれる、権限オブジェクトの配列。たとえば、W&B の Runs に対する削除操作の権限オブジェクトには、`name` として `run:delete` が設定されます。 |
| inheritedFrom | 文字列 | カスタムロールが継承する定義済みのロール。`member` または `viewer` のいずれかになります。 |
- **リクエストの例**:

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

- **レスポンスの例**:

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

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: DELETE
- **説明**: W&B organization 内のカスタムロールを削除します。**使用には注意してください**。カスタムロールが継承した定義済みのロールが、操作前にカスタムロールを割り当てられていたすべての Users に割り当てられるようになりました。
- **リクエストの例**:

```bash
DELETE /scim/Roles/abc
```

- **レスポンスの例**:

```bash
(Status 204)
```

### カスタムロールの権限を更新

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: PATCH
- **説明**: W&B organization のカスタムロールで、カスタム権限を追加または削除します。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| operations | オブジェクト配列 | 操作オブジェクトの配列 |
| op | 文字列 | 操作オブジェクト内の操作のタイプ。`add` または `remove` のいずれかになります。 |
| path | 文字列 | 操作オブジェクト内の静的フィールド。許可される value は `permissions` のみです。 |
| value | オブジェクト配列 | 各オブジェクトに `w&bobject:operation` の形式の value を持つ `name` 文字列フィールドが含まれる、権限オブジェクトの配列。たとえば、W&B の Runs に対する削除操作の権限オブジェクトには、`name` として `run:delete` が設定されます。 |
- **リクエストの例**:

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

- **レスポンスの例**:

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

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: PUT
- **説明**: W&B organization のカスタムロールの名前、説明、または継承されたロールを更新します。この操作は、カスタムロールの既存の（つまり、継承されていない）カスタム権限には影響しません。
- **サポートされているフィールド**:

| フィールド | タイプ | 必須 |
| --- | --- | --- |
| name | 文字列 | カスタムロールの名前 |
| description | 文字列 | カスタムロールの説明 |
| inheritedFrom | 文字列 | カスタムロールが継承する定義済みのロール。`member` または `viewer` のいずれかになります。 |
- **リクエストの例**:

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

- **レスポンスの例**:

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