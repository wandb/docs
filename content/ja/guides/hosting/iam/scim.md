---
title: ユーザー、グループ、およびロールを SCIM で管理する
menu:
  default:
    identifier: ja-guides-hosting-iam-scim
    parent: identity-and-access-management-iam
weight: 4
---

{{% alert %}}
[SCIM を実際に使用するビデオを見る](https://www.youtube.com/watch?v=Nw3QBqV0I-o) (12分)
{{% /alert %}}

クロスドメイン・アイデンティティ管理システム (SCIM) API は、インスタンスまたは組織の管理者が W&B 組織内のユーザー、グループ、およびカスタムロールを管理できるようにします。SCIM グループは W&B のチームにマップされます。

SCIM API は `<host-url>/scim/` でアクセス可能であり、[RC7643 プロトコル](https://www.rfc-editor.org/rfc/rfc7643)に見られるフィールドのサブセットを持つ `/Users` と `/Groups` エンドポイントをサポートしています。さらに、公式の SCIM スキーマの一部ではない `/Roles` エンドポイントを含んでいます。W&B は `/Roles` エンドポイントを追加して、W&B 組織でのカスタムロールの自動管理をサポートしています。

{{% alert %}}
複数の Enterprise [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) 組織の管理者である場合、SCIM API リクエストが送信される組織を設定する必要があります。プロファイル画像をクリックし、**User Settings** をクリックします。設定名は **Default API organization** です。これは、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、[Self-managed instances]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})、および [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) を含むすべてのホスティングオプションで必要です。SaaS Cloud では、組織の管理者は SCIM API リクエストが正しい組織に行くように、ユーザー設定でデフォルトの組織を設定する必要があります。

選択したホスティングオプションに応じて、このページの例で使用される `<host-url>` プレースホルダーの値が決まります。

さらに、例では `abc` や `def` のようなユーザー ID が使用されています。実際のリクエストとレスポンスでは、ユーザー ID はハッシュ化された値を持っています。
{{% /alert %}}

## 認証

組織またはインスタンスの管理者は、自分の API キーを使用した基本認証を利用して SCIM API にアクセスできます。HTTP リクエストの `Authorization` ヘッダーを文字列 `Basic` の後にスペースを置き、その後のベース64エンコードされた文字列を `username:API-KEY` 形式に設定します。つまり、ユーザー名と API キーを `:` 文字で区切った後、その結果をベース64エンコードします。例えば、`demo:p@55w0rd` として認証するには、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` であるべきです。

## ユーザーリソース

SCIM ユーザーリソースは W&B のユーザーにマップされます。

### ユーザーの取得

- **エンドポイント:** **`<host-url>/scim/Users/{id}`**
- **メソッド**: GET
- **説明**: ユーザーの一意の ID を提供することにより、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) 組織または [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内の特定のユーザーの情報を取得します。
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

### ユーザーのリスト

- **エンドポイント:** **`<host-url>/scim/Users`**
- **メソッド**: GET
- **説明**: [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) 組織または [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内のすべてのユーザーのリストを取得します。
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

| フィールド | 型 | 必要 |
| --- | --- | --- |
| emails | Multi-Valued Array | Yes (必ず `primary` email を設定してください) |
| userName | 文字列型 | Yes |
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
- **説明**: ユーザーの一意の ID を提供することにより、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) 組織または [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスから完全にユーザーを削除します。必要に応じて [Create user]({{< relref path="#create-user" lang="ja" >}}) API を使用して組織またはインスタンスに再度ユーザーを追加してください。
- **リクエスト例**:

{{% alert %}}
一時的にユーザーを無効化するには、`PATCH` エンドポイントを使用する [Deactivate user]({{< relref path="#deactivate-user" lang="ja" >}}) API を参照してください。
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
- **説明**: ユーザーの一意の ID を提供することにより、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内で一時的にユーザーを無効化します。必要に応じて [Reactivate user]({{< relref path="#reactivate-user" lang="ja" >}}) API を使用してユーザーを再有効化します。
- **サポートされているフィールド**:

| フィールド | 型 | 必要 |
| --- | --- | --- |
| op | 文字列型 | 操作のタイプ。許可される唯一の値は `replace` です。 |
| value | オブジェクト型 | オブジェクト `{"active": false}` を示し、ユーザーが無効化されるべきことを示します。 |

{{% alert %}}
[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、ユーザーの無効化および再有効化の操作はサポートされていません。
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
この操作はユーザーオブジェクトを返します。

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
- **説明**: ユーザーの一意の ID を提供することにより、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス内で無効化されたユーザーを再有効化します。
- **サポートされているフィールド**:

| フィールド | 型 | 必要 |
| --- | --- | --- |
| op | 文字列型 | 操作のタイプ。許可される唯一の値は `replace` です。 |
| value | オブジェクト型 | オブジェクト `{"active": true}` を示し、ユーザーが再有効化されるべきことを示します。 |

{{% alert %}}
[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、ユーザーの無効化および再有効化の操作はサポートされていません。
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
この操作はユーザーオブジェクトを返します。

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
- **説明**: 組織レベルのロールをユーザーに割り当てます。このロールは、ここで説明されているように `admin`、`viewer` または `member` のいずれかになります ([こちらを参照]({{< relref path="access-management/manage-organization.md#invite-a-user" lang="ja" >}}))。 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、ユーザー設定で SCIM API の正しい組織を設定することを確認してください。
- **サポートされているフィールド**:

| フィールド | 型 | 必要 |
| --- | --- | --- |
| op | 文字列型 | 操作のタイプ。許可される唯一の値は `replace` です。 |
| path | 文字列型 | ロール割り当て操作が影響を及ぼすスコープ。許可される唯一の値は `organizationRole` です。 |
| value | 文字列型 | ユーザーに割り当てる予定の定義済みの組織レベルのロール。それは `admin`、`viewer` または `member` のいずれかです。このフィールドは定義済みロールに対して大文字小文字を区別しません。 |
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
            "value": "admin" // ユーザーの組織スコープのロールを admin に設定
        }
    ]
}
```

- **レスポンス例**:
この操作はユーザーオブジェクトを返します。

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

### チームレベルのロールをユーザーに割り当てる

- **エンドポイント**: **`<host-url>/scim/Users/{id}`**
- **メソッド**: PATCH
- **説明**: チームレベルのロールをユーザーに割り当てます。このロールは、ここで説明されているように `admin`、`viewer`、`member` またはカスタムロールのいずれかになります ([こちらを参照]({{< relref path="access-management/manage-organization.md#assign-or-update-a-team-members-role" lang="ja" >}}))。 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では、ユーザー設定で SCIM API の正しい組織を設定することを確認してください。
- **サポートされているフィールド**:

| フィールド | 型 | 必要 |
| --- | --- | --- |
| op | 文字列型 | 操作のタイプ。許可される唯一の値は `replace` です。 |
| path | 文字列型 | ロール割り当て操作が影響を及ぼすスコープ。許可される唯一の値は `teamRoles` です。 |
| value | オブジェクト配列型 | 1つのオブジェクトを持つ配列で、そのオブジェクトは `teamName` と `roleName` 属性を持ちます。`teamName` はユーザーがそのロールを持つチームの名前であり、`roleName` は `admin`、`viewer`、`member` またはカスタムロールのいずれかです。このフィールドは定義済みロールに対して大文字小文字を区別しませんが、カスタムロールに対しては区別します。 |
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
                    "roleName": "admin", // 定義済みロールのロール名は大文字小文字を区別しませんが、カスタムロールでは区別します
                    "teamName": "team1" // チームteam1でのユーザーのロールをadminに設定
                }
            ]
        }
    ]
}
```

- **レスポンス例**:
この操作はユーザーオブジェクトを返します。

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

## グループリソース

SCIM グループリソースは W&B のチームにマップされます。つまり、W&B デプロイメントで SCIM グループを作成すると W&B チームが作成されます。その他のグループエンドポイントにも同様です。

### チームの取得

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

### チームのリスト

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

| フィールド | 型 | 必要 |
| --- | --- | --- |
| displayName | 文字列型 | Yes |
| members | Multi-Valued Array | Yes (`value` サブフィールドが必要で、ユーザー ID にマップされます) |
- **リクエスト例**:

`wandb-support` というチームを作成し、そのメンバーとして `dev-user2` を設定します。

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
- **サポートされている操作**: メンバーの`追加`、メンバーの`削除`
- **リクエスト例**:

`wandb-devs` に `dev-user2` を追加する

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

- SCIM API では、チームには追加データがリンクされているため、現在チームの削除はサポートされていません。削除を確認するにはアプリからチームを削除してください。

## ロールリソース

SCIM ロールリソースは W&B のカスタムロールにマップされます。前述したように、`/Roles` エンドポイントは公式 SCIM スキーマの一部ではありません。W&B は `W&B` 組織内のカスタムロールの自動管理をサポートするために `/Roles` エンドポイントを追加しています。

### カスタムロールの取得

- **エンドポイント:** **`<host-url>/scim/Roles/{id}`**
- **メソッド**: GET
- **説明**: ロールの一意の ID を提供し、カスタムロールの情報を取得します。
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
    "inheritedFrom": "member", // 定義済みロールを示します
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
            "isInherited": true // member 定義済みロールから継承された
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

- **エンドポイント:** **`<host-url>/scim/Roles`**
- **メソッド**: GET
- **説明**: W&B 組織のすべてのカスタムロールの情報を取得します
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
            "inheritedFrom": "member", // 定義済みロールからカスタムロールが継承されていることを示します
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
                    "isInherited": true // member 定義済みロールから継承された
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
            "inheritedFrom": "viewer", // 定義済みロールからカスタムロールが継承されていることを示します
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
                    "isInherited": true // viewer 定義済みロールから継承された
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
- **説明**: W&B 組織内で新しいカスタムロールを作成します。
- **サポートされているフィールド**:

| フィールド | 型 | 必要 |
| --- | --- | --- |
| name | 文字列型 | カスタムロールの名前 |
| description | 文字列型 | カスタムロールの説明 |
| permissions | オブジェクト配列型 | 各オブジェクトが `name` 文字列フィールドを含む許可オブジェクトの配列で、そのフィールドは `w&bobject:operation` の形式を持ちます。例えば、W&B Run に対する削除操作の許可オブジェクトは `name` を `run:delete` として持ちます。 |
| inheritedFrom | 文字列型 | カスタムロールが継承する定義済みロール。それは `member` または `viewer` のいずれかになります。 |
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
    "inheritedFrom": "member", // 定義済みロールを示します
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
            "isInherited": true // member 定義済みロールから継承された
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
- **説明**: W&B 組織内のカスタムロールを削除します。**慎重に使用してください。** カスタムロールから継承される定義済みロールは、操作前にカスタムロールに割り当てられていたすべてのユーザーに再び割り当てられます。
- **リクエスト例**:

```bash
DELETE /scim/Roles/abc
```

- **レスポンス例**:

```bash
(Status 204)
```

### カスタムロールの権限の更新

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: PATCH
- **説明**: W&B 組織内のカスタムロールにカスタム権限を追加または削除します。
- **サポートされているフィールド**:

| フィールド | 型 | 必要 |
| --- | --- | --- |
| operations | オブジェクト配列型 | 操作オブジェクトの配列 |
| op | 文字列型 | 操作オブジェクト内の操作のタイプ。それは `add` または `remove` のいずれかになります。 |
| path | 文字列型 | 操作オブジェクト内の静的フィールド。許可される唯一の値は `permissions` です。 |
| value | オブジェクト配列型 | 各オブジェクトが `name` 文字列フィールドを含む許可オブジェクトの配列で、そのフィールドは `w&bobject:operation` の形式を持ちます。例えば、W&B Run に対する削除操作の許可オブジェクトは `name` を `run:delete` として持ちます。 |
- **リクエスト例**:

```bash
PATCH /scim/Roles/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add", // 操作のタイプを示し、他の可能な値は `remove`
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
    "inheritedFrom": "member", // 定義済みロールを示します
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
            "isInherited": true // member 定義済みロールから継承された
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 更新前に管理者によって追加された既存のカスタム権限
        },
        {
            "name": "project:delete",
            "isInherited": false // 更新の一部として管理者によって追加された新規のカスタム権限
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
- **説明**: W&B 組織内のカスタムロールの名前、説明、または継承ロールを更新します。この操作は、既存の、つまり非継承のカスタム権限には影響しません。
- **サポートされているフィールド**:

| フィールド | 型 | 必要 |
| --- | --- | --- |
| name | 文字列型 | カスタムロールの名前 |
| description | 文字列型 | カスタムロールの説明 |
| inheritedFrom | 文字列型 | カスタムロールが継承する定義済みロール。それは `member` または `viewer` のいずれかになります。 |
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
    "description": "A sample custom role for example but now based on viewer", // リクエストに応じて説明を変更
    "id": "Um9sZTo3",
    "inheritedFrom": "viewer", // リクエストに応じて変更された定義済みロールを示します
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
            "isInherited": true // viewer 定義済みロールから継承された
        },
        ... // 更新後に member 定義済みロールにあるが viewer にはない権限は継承されません
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