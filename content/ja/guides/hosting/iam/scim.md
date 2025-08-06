---
title: SCIM で ユーザー、グループ、ロールを管理する
menu:
  default:
    identifier: scim
    parent: identity-and-access-management-iam
weight: 4
---

{{% alert %}}
[SCIM を実際に操作するデモ動画はこちら](https://www.youtube.com/watch?v=Nw3QBqV0I-o)（12分）
{{% /alert %}}

## 概要

System for Cross-domain Identity Management (SCIM) API を利用すると、インスタンスや組織の管理者は W&B 組織内のユーザー、グループ、カスタムロールを管理できます。SCIM グループは W&B の Teams（チーム）にマッピングされます。

SCIM API へのアクセスは `<host-url>/scim/` で提供されており、`/Users` と `/Groups` エンドポイントでは [RC7643プロトコル](https://www.rfc-editor.org/rfc/rfc7643) の一部フィールドのみをサポートしています。さらに、公式 SCIM スキーマには含まれていない `/Roles` エンドポイントも提供されており、W&B 組織内のカスタムロールの自動管理を実現します。

{{% alert %}}
複数の Enterprise [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) 組織の管理者の場合、SCIM API リクエストを送る組織を設定する必要があります。プロフィール画像をクリックし、**User Settings** を開いてください。この設定は **Default API organization** という名前になっています。  
この設定はすべてのホスティングオプション、[Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}})、[セルフマネージドインスタンス]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}})、[SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}})で必要です。SaaS Cloud では、組織管理者がユーザー設定からデフォルトの組織を設定し、SCIM API リクエストが正しい組織に送信されるようにします。

どのホスティングオプションを選択しているかによって、このページの例で使われている `<host-url>` の値が変わります。

また、例ではユーザー ID として `abc` や `def` のような値を使っていますが、実際のリクエスト・レスポンスではハッシュ化されたユーザー ID になります。
{{% /alert %}}

## 認証

SCIM API への認証方法は 2 通りあります。

### ユーザー

組織またはインスタンスの管理者は、自身の APIキー を使ったベーシック認証で SCIM API に アクセス できます。HTTP リクエストの `Authorization` ヘッダには `Basic` という文字列とスペースの後に、`username:API-KEY` という文字列を base-64 エンコードしたものを指定してください。つまり、ユーザー名と APIキー を `:` で区切って連結し、それを base-64 エンコードします。  
例）`demo:p@55w0rd` で認証する場合、ヘッダーの内容は `Authorization: Basic ZGVtbzpwQDU1dzByZA==` となります。

### サービスアカウント

`admin` ロールを持つ組織サービスアカウントは SCIM API に アクセス できます。ユーザー名は空欄とし、APIキー のみを利用します。サービスアカウントの APIキー は、組織ダッシュボードの **Service account** タブで確認できます。[Organization-scoped service accounts]({{< relref "/guides/hosting/iam/authentication/service-accounts.md/#organization-scoped-service-accounts" >}}) を参照してください。

HTTP リクエストの `Authorization` ヘッダには `Basic :API-KEY` を base-64 エンコードしたものを指定します（先頭のコロン `:`は必須、ユーザー名は不要です）。  
例）APIキー `sa-p@55w0rd` で認証する場合、`Authorization: Basic OnNhLXBANTV3MHJk` となります。

## ユーザー管理

SCIM のユーザーリソースは W&B ユーザーに対応します。これらのエンドポイントを使って、自組織のユーザー管理が行えます。

### ユーザー情報の取得

組織内の特定ユーザー情報を取得します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: GET

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | Yes | ユーザーの一意のID |

#### 例

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

### ユーザー一覧の取得

組織内のすべてのユーザー一覧を取得します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users`
- **メソッド**: GET

#### 例

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

### ユーザーの作成

組織内に新しいユーザーを作成します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users`
- **メソッド**: POST

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| emails | array | Yes | メール情報オブジェクトの配列。プライマリメールの指定が必要 |
| userName | string | Yes | 新しいユーザーのユーザー名 |

#### 例

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

#### レスポンス

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

### ユーザーの削除

{{% alert color="warning" title="管理者アクセスの維持について" %}}
インスタンスもしくは組織に、常に最低1人は admin ユーザーがいることを保証してください。もし管理者がいなくなると、組織の W&B アカウントの設定や保守ができなくなります。SCIM や他の自動化プロセスでユーザーを削除した際、うっかり最後の管理者を削除しないよう注意してください。

運用フロー策定サポートや管理者アクセスの復元は [support](mailto:support@wandb.com) までご相談ください。
{{% /alert %}}

組織から完全にユーザーを削除します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: DELETE

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | Yes | 削除対象ユーザーの一意のID |

#### 例

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
一時的な無効化は [Deactivate user](#deactivate-user) API（PATCH エンドポイント）を利用してください。
{{% /alert %}}

### ユーザーの一時無効化

組織内のユーザーを一時的に無効化します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | Yes | 無効化するユーザーの一意のID |
| op | string | Yes | "replace" を指定してください |
| value | object | Yes | `{"active": false}` を含むオブジェクト |

{{% alert %}}
ユーザーの無効化・再有効化操作は [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) ではサポートされていません。
{{% /alert %}}

#### 例

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

### ユーザーの再有効化

一時無効化されたユーザーを再度有効化します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | Yes | 再有効化するユーザーの一意のID |
| op | string | Yes | "replace" を指定してください |
| value | object | Yes | `{"active": true}` を含むオブジェクト |

{{% alert %}}
ユーザーの無効化・再有効化操作は [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) ではサポートされていません。
{{% /alert %}}

#### 例

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

### 組織ロールの割り当て

ユーザーに組織レベルのロールを付与します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | Yes | ユーザーの一意のID |
| op | string | Yes | "replace" を指定してください |
| path | string | Yes | "organizationRole" を指定 |
| value | string | Yes | ロール名（"admin" または "member"）|

{{% alert %}}
`viewer` ロールは非推奨で、UI からは設定できません。SCIM で viewer を割り当てようとした場合、自動的に member ロールが割り当てられます。ユーザーには可能であれば Models と Weave のシートが割り当てられます。割り当て上限を超える場合は `Seat limit reached` エラーが記録されます。**Registry** を利用する組織の場合、組織レベルで参照可能な Registry には viewer 権限で自動付与されます。
{{% /alert %}}

#### 例

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

### チームロールの割り当て

ユーザーにチームレベルのロールを付与します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | Yes | ユーザーの一意のID |
| op | string | Yes | "replace" を指定してください |
| path | string | Yes | "teamRoles" を指定 |
| value | array | Yes | `teamName` と `roleName` の入った配列 |

#### 例

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

## Group リソース

SCIM のグループリソースは W&B の Teams（チーム）に対応します。SCIM でグループを作成すると W&B デプロイメントにチームが作られます。他のグループ用エンドポイントも同様です。

### チーム情報の取得

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: GET
- **説明**: チームの一意のIDを指定して、チーム情報を取得します。
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

### チーム一覧の取得

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
- **対応フィールド**:

| フィールド | 型 | 必須 |
| --- | --- | --- |
| displayName | String | Yes |
| members | Multi-Valued Array | Yes（`value` サブフィールドは必須、ユーザーIDとしてマッピング）|

- **リクエスト例**:

`wandb-support` というチームに `dev-user2` をメンバーとして追加する例。

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
- **説明**: 既存チームのメンバーリストを更新します。
- **対応操作**: メンバーの `add`、`remove`

{{% alert %}}
remove 操作は RFC 7644 SCIM プロトコル仕様に準拠しています。`members[value eq "{user_id}"]` というフィルター構文で特定ユーザーを削除、`members` で全ユーザーを削除できます。
{{% /alert %}}

{{% alert color="info" %}}
`{team_id}` は実際のチーム ID、`{user_id}` は実際のユーザー ID で置き換えてください。
{{% /alert %}}

**チームへのユーザー追加**

`wandb-devs` へ `dev-user2` を追加する例：

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

**チームから特定ユーザー削除**

`wandb-devs` から `dev-user2` を削除する例：

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

**すべてのユーザーをチームから削除**

`wandb-devs` から全ユーザーを削除する例：

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

### チームの削除

- SCIM API では現状、Teams の削除はサポートされていません（チームに紐付く追加データがあるため）。アプリ上で削除操作を実施し、全データ削除の意思確認を行ってください。

## Role リソース

SCIM のロールリソースは W&B のカスタムロールに対応します。前述のとおり `/Roles` エンドポイントは正式な SCIM スキーマには含まれず、W&B が組織内カスタムロールの自動管理のために追加実装しています。

### カスタムロール情報の取得

- **エンドポイント:** **`<host-url>/scim/Roles/{id}`**
- **メソッド**: GET
- **説明**: 一意のロールIDを指定してカスタムロール情報を取得します。
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
    "inheritedFrom": "member", // 標準ロールから継承していることを示す
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
            "isInherited": true // member 標準ロールから継承
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // adminが追加したカスタム権限
        }
    ],
    "schemas": [
        ""
    ]
}
```

### カスタムロール一覧の取得

- **エンドポイント:** **`<host-url>/scim/Roles`**
- **メソッド**: GET
- **説明**: W&B 組織内のすべてのカスタムロール情報を取得します。
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
            "inheritedFrom": "member", // カスタムロールが継承する標準ロール
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
                    "isInherited": true // member 標準ロールから継承
                },
                ...
                ...
                {
                    "name": "project:update",
                    "isInherited": false // adminが追加したカスタム権限
                }
            ],
            "schemas": [
                ""
            ]
        },
        {
            "description": "Another sample custom role for example",
            "id": "Um9sZToxMg==",
            "inheritedFrom": "viewer", // カスタムロールが継承する標準ロール
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
                    "isInherited": true // viewer 標準ロールから継承
                },
                ...
                ...
                {
                    "name": "run:stop",
                    "isInherited": false // adminが追加したカスタム権限
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
- **説明**: W&B 組織に新しいカスタムロールを作成します。
- **対応フィールド**:

| フィールド | 型 | 必須 |
| --- | --- | --- |
| name | String | カスタムロール名 |
| description | String | カスタムロールの説明 |
| permissions | Object array | 各オブジェクトは `w&bobject:operation` 形式の `name` フィールド（例：run:delete で Run の削除権限） |
| inheritedFrom | String | 継承元の標準ロール。`member` または `viewer` |

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
    "inheritedFrom": "member", // 標準ロールから継承していることを示す
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
            "isInherited": true // member 標準ロールから継承
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // adminが追加したカスタム権限
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
- **説明**: W&B 組織のカスタムロールを削除します。**操作には十分ご注意ください。** カスタムロールを割り当てられていたユーザーには自動的に継承元の標準ロールが割り当てられます。
- **リクエスト例**:

```bash
DELETE /scim/Roles/abc
```
