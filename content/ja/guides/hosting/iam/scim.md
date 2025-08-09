---
title: SCIM を使ってユーザー、グループ、ロールを管理する
menu:
  default:
    identifier: ja-guides-hosting-iam-scim
    parent: identity-and-access-management-iam
weight: 4
---

{{% alert %}}
[SCIM の動作デモ動画を見る](https://www.youtube.com/watch?v=Nw3QBqV0I-o)（約12分）
{{% /alert %}}

## 概要

System for Cross-domain Identity Management（SCIM）APIは、インスタンスまたは組織の管理者が W&B 組織内のユーザー、グループ、カスタムロールを管理できる API です。SCIM のグループは W&B の Teams にマッピングされます。

SCIM API は `<host-url>/scim/` で利用でき、`/Users` および `/Groups` エンドポイントでは [RC7643 プロトコル](https://www.rfc-editor.org/rfc/rfc7643) に記載されているフィールドのサブセットがサポートされています。さらに、公式 SCIM スキーマにはない `/Roles` エンドポイントも含まれています。これは W&B 組織でカスタムロールの自動管理を可能にするために追加されています。

{{% alert %}}
複数の Enterprise [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) 組織の管理者は、SCIM API リクエストを送信する対象の組織を設定する必要があります。プロフィール画像をクリックし、**ユーザー設定** をクリックします。設定名は **デフォルト API 組織** です。この設定は、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、[セルフマネージドインスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) など、すべてのホスティングオプションで必須です。SaaS Cloud では、組織管理者がユーザー設定でデフォルト組織を設定することで、SCIM API リクエストが正しい組織に向かうようにしてください。

選択したホスティングオプションによって、ページ内で例示される `<host-url>` プレースホルダーの値が異なります。

また、例ではユーザーIDとして `abc` や `def` などを使用していますが、実際のリクエストやレスポンスではハッシュ化されたユーザーIDが使われます。
{{% /alert %}}

## 認証

SCIM API へのアクセスには2通りの認証方法があります。

### ユーザー

組織またはインスタンスの管理者は、自身の APIキー を使ったベーシック認証で SCIM API にアクセスできます。HTTP リクエストの `Authorization` ヘッダーには `Basic` の後に半角スペース + `username:API-KEY` の形式を base64 エンコードした文字列を設定してください。つまり、ユーザー名と APIキー をコロン `:` でつなぎ、それを base64 エンコードします。例として、`demo:p@55w0rd` で認証する場合、ヘッダーは `Authorization: Basic ZGVtbzpwQDU1dzByZA==` となります。

### サービスアカウント

`admin` ロールを持つ組織のサービスアカウントも SCIM API へのアクセスが可能です。この場合、ユーザー名は空欄とし、APIキー のみを使用します。サービスアカウント用の APIキー は組織ダッシュボードの **サービスアカウント** タブで確認できます。詳しくは [Organization-scoped service accounts]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md/#organization-scoped-service-accounts" lang="ja" >}}) を参照してください。

HTTP リクエストの `Authorization` ヘッダーには、`Basic` の後に半角スペース + `:API-KEY` という形式（ユーザー名なしのコロンで始まる）を base64 エンコードした文字列を指定します。例えば、APIキー が `sa-p@55w0rd` の場合、ヘッダーは `Authorization: Basic OnNhLXBANTV3MHJk` となります。

## ユーザー管理

SCIM のユーザーリソースは W&B のユーザーに対応します。これらのエンドポイントを使って、組織内のユーザー管理ができます。

### ユーザー情報の取得

組織内の特定ユーザーの情報を取得します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: GET

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | はい | ユーザーの一意なID |

#### 例

{{< tabpane text=true >}}
{{% tab header="ユーザー取得リクエスト" %}}
```bash
GET /scim/Users/abc
```
{{% /tab %}}
{{% tab header="ユーザー取得レスポンス" %}}
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

### ユーザー一覧取得

組織内のすべてのユーザーのリストを取得します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users`
- **メソッド**: GET

#### 例

{{< tabpane text=true >}}
{{% tab header="ユーザー一覧リクエスト" %}}
```bash
GET /scim/Users
```
{{% /tab %}}
{{% tab header="ユーザー一覧レスポンス" %}}
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

組織に新しいユーザーを作成します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users`
- **メソッド**: POST

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| emails | array | はい | メールオブジェクトの配列。プライマリメールが必須 |
| userName | string | はい | 新しいユーザーのユーザー名 |

#### 例

{{< tabpane text=true >}}
{{% tab header="ユーザー作成リクエスト（専用・セルフマネージド）" %}}
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
{{% tab header="ユーザー作成リクエスト（マルチテナント）" %}}
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
{{% tab header="ユーザー作成レスポンス（専用・セルフマネージド）" %}}
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
{{% tab header="ユーザー作成レスポンス（マルチテナント）" %}}
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

{{% alert color="warning" title="管理者アクセスの維持" %}}
インスタンスまたは組織には、常に少なくとも1名以上の管理者ユーザーが存在する必要があります。これを満たさない場合、いずれのユーザーも組織の W&B アカウントを設定・管理できなくなります。SCIM やその他の自動化プロセスで W&B からユーザーを削除する場合、最後の管理者が誤って削除される可能性がありますのでご注意ください。

運用手順の作成や管理者アクセスの復元については [サポート](mailto:support@wandb.com) までご相談ください。
{{% /alert %}}

ユーザーを完全に組織から削除します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: DELETE

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | はい | 削除対象ユーザーの一意なID |

#### 例

{{< tabpane text=true >}}
{{% tab header="ユーザー削除リクエスト" %}}
```bash
DELETE /scim/Users/abc
```
{{% /tab %}}
{{% tab header="ユーザー削除レスポンス" %}}
```bash
(Status 204)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
一時的な無効化の場合は、`PATCH` エンドポイントを使用する [ユーザーの一時停止](#deactivate-user) API をご利用ください。
{{% /alert %}}

### ユーザーの一時停止

組織内でユーザーを一時的に無効化します。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | はい | 一時停止するユーザーの一意なID |
| op | string | はい | `"replace"` を指定する必要があります |
| value | object | はい | `{"active": false}` のオブジェクト |

{{% alert %}}
[ SaaS Cloud ]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではユーザーの一時停止・再有効化操作はサポートされていません。
{{% /alert %}}

#### 例

{{< tabpane text=true >}}
{{% tab header="ユーザー一時停止リクエスト" %}}
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
{{% tab header="ユーザー一時停止レスポンス" %}}
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

一時的に停止されたユーザーを再度有効にします。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | はい | 再有効化するユーザーの一意なID |
| op | string | はい | `"replace"` を指定する必要があります |
| value | object | はい | `{"active": true}` のオブジェクト |

{{% alert %}}
[ SaaS Cloud ]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではユーザーの一時停止・再有効化操作はサポートされていません。
{{% /alert %}}

#### 例

{{< tabpane text=true >}}
{{% tab header="ユーザー再有効化リクエスト" %}}
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
{{% tab header="ユーザー再有効化レスポンス" %}}
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

ユーザーに組織レベルのロールを割り当てます。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | はい | ユーザーの一意なID |
| op | string | はい | `"replace"` を指定する必要があります |
| path | string | はい | `"organizationRole"` を指定する必要があります |
| value | string | はい | ロール名（"admin" または "member"） |

{{% alert %}}
`viewer` ロールは廃止されており UI からは設定できません。SCIM で `viewer` ロールを割り当てようとすると自動的に `member` ロールが割り当てられます。可能であればモデルと Weave のシートも自動的に割り当てられます。ただしシート上限に達した場合は `Seat limit reached` エラーが記録されます。**レジストリ** を利用している組織では、ユーザーは組織レベルで公開されているレジストリに対して自動で `viewer` ロールが割り当てられます。
{{% /alert %}}

#### 例

{{< tabpane text=true >}}
{{% tab header="組織ロール割り当てリクエスト" %}}
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
{{% tab header="組織ロール割り当てレスポンス" %}}
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

ユーザーにチームレベルのロールを割り当てます。

#### エンドポイント
- **URL**: `<host-url>/scim/Users/{id}`
- **メソッド**: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| id | string | はい | ユーザーの一意なID |
| op | string | はい | `"replace"` を指定する必要があります |
| path | string | はい | `"teamRoles"` を指定する必要があります |
| value | array | はい | `teamName` と `roleName` を持つオブジェクト配列 |

#### 例

{{< tabpane text=true >}}
{{% tab header="チームロール割り当てリクエスト" %}}
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
{{% tab header="チームロール割り当てレスポンス" %}}
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

## グループリソース

SCIM のグループリソースは W&B の Teams にマッピングされます。つまり、SCIM でグループを作成すると W&B の Team も作成されます。他のグループエンドポイントも同様です。

### チーム情報の取得

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: GET
- **説明**: チームの一意なIDでチーム情報を取得します。
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

### チーム一覧取得

- **エンドポイント**: **`<host-url>/scim/Groups`**
- **メソッド**: GET
- **説明**: チームの一覧を取得します。
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
- **対応項目**:

| フィールド | 型 | 必須 |
| --- | --- | --- |
| displayName | String | はい |
| members | Multi-Valued Array | はい（`value` サブフィールドは必須。ユーザーIDを指定） |
- **リクエスト例**:

`dev-user2` をメンバーとして持つ `wandb-support` チームの作成例

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

- **エンドポイント**: **`<host-url>/scim/Groups/{id}`**
- **メソッド**: PATCH
- **説明**: 既存チームのメンバーリストを更新します。
- **対応操作**: メンバーの `add`、`remove`

{{% alert %}}
削除操作は RFC 7644 SCIM プロトコル仕様に従います。`members[value eq "{user_id}"]` で特定ユーザーを削除、`members` で全ユーザー削除となります。
{{% /alert %}}

{{% alert color="info" %}}
`{team_id}` は実際のチームIDに、`{user_id}` は実際のユーザーIDに置き換えてご利用ください。
{{% /alert %}}

**チームへのユーザー追加**

`wandb-devs` へ `dev-user2` を追加する例：

{{< tabpane text=true >}}
{{% tab header="リクエスト" %}}
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
{{% tab header="レスポンス" %}}
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

**チームから特定ユーザーを削除**

`wandb-devs` から `dev-user2` を削除する例：

{{< tabpane text=true >}}
{{% tab header="リクエスト" %}}
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
{{% tab header="レスポンス" %}}
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

**チームから全ユーザーを削除**

`wandb-devs` から全ユーザーを削除する例：

{{< tabpane text=true >}}
{{% tab header="リクエスト" %}}
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
{{% tab header="レスポンス" %}}
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

### チーム削除

- チーム削除は SCIM API では現在サポートされていません（Teams の情報に追加データが紐付いているため）。全て削除したい場合はアプリからチームを削除してください。

## ロールリソース

SCIM のロールリソースは W&B のカスタムロールに対応します。上述の通り、`/Roles` エンドポイントは公式 SCIM スキーマ上にはありませんが、W&B がカスタムロールの自動管理を実現するために追加しているものです。

### カスタムロール情報の取得

- **エンドポイント:** **`<host-url>/scim/Roles/{id}`**
- **メソッド**: GET
- **説明**: ロールの一意なIDでカスタムロール情報を取得します。
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
    "inheritedFrom": "member", // 継承元のプリセットロール
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
            "isInherited": true // member プリセットロールから継承
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 管理者が追加したカスタム権限
        }
    ],
    "schemas": [
        ""
    ]
}
```

### カスタムロール一覧取得

- **エンドポイント:** **`<host-url>/scim/Roles`**
- **メソッド**: GET
- **説明**: W&B 組織内の全カスタムロール情報を取得します。
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
            "inheritedFrom": "member", // カスタムロール継承元のプリセットロール
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
                    "isInherited": true // member プリセットロールから継承
                },
                ...
                ...
                {
                    "name": "project:update",
                    "isInherited": false // 管理者が追加したカスタム権限
                }
            ],
            "schemas": [
                ""
            ]
        },
        {
            "description": "Another sample custom role for example",
            "id": "Um9sZToxMg==",
            "inheritedFrom": "viewer", // カスタムロール継承元のプリセットロール
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
                    "isInherited": true // viewer プリセットロールから継承
                },
                ...
                ...
                {
                    "name": "run:stop",
                    "isInherited": false // 管理者が追加したカスタム権限
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

### カスタムロール作成

- **エンドポイント**: **`<host-url>/scim/Roles`**
- **メソッド**: POST
- **説明**: W&B 組織に新しいカスタムロールを作成します。
- **対応項目**:

| フィールド | 型 | 必須 |
| --- | --- | --- |
| name | String | カスタムロール名 |
| description | String | カスタムロールの説明 |
| permissions | Object array | 権限オブジェクトの配列。各オブジェクトの `name` は `w&bobject:operation` 形式の文字列。例：W&B の run 削除は `name` に `run:delete` を指定。|
| inheritedFrom | String | 継承元プリセットロール。`member` または `viewer` いずれか。|
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
    "inheritedFrom": "member", // 継承元プリセットロール
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
            "isInherited": true // member プリセットロールから継承
        },
        ...
        ...
        {
            "name": "project:update",
            "isInherited": false // 管理者が追加したカスタム権限
        }
    ],
    "schemas": [
        ""
    ]
}
```

### カスタムロール削除

- **エンドポイント**: **`<host-url>/scim/Roles/{id}`**
- **メソッド**: DELETE
- **説明**: W&B 組織内のカスタムロールを削除します。**利用は慎重に行ってください。** 削除後、そのカスタムロールを割り当てられていたユーザーには元のプリセットロールが自動的に割り当てられます。
- **リクエスト例**:

```bash
DELETE /scim/Roles/abc
```
