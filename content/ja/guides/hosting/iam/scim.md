---
title: SCIM で ユーザー、グループ、ロールを管理する
menu:
  default:
    identifier: ja-guides-hosting-iam-scim
    parent: identity-and-access-management-iam
weight: 4
---

{{% alert %}}
[SCIM の実演動画を見る](https://www.youtube.com/watch?v=Nw3QBqV0I-o)（12 分）
{{% /alert %}}

## 概要

System for Cross-domain Identity Management（SCIM）API は、インスタンスまたは組織の管理者が W&B の組織内の User、グループ、カスタムロールを管理できるようにする API です。SCIM のグループは W&B の Team に対応します。

W&B の SCIM API は Okta を含む主要なアイデンティティプロバイダに対応しており、ユーザーの自動プロビジョニング／プロビジョニング解除を実現します。Okta やその他の IdP での SSO 設定については、[SSO ドキュメント]({{< relref path="/guides/hosting/iam/authentication/sso.md" lang="ja" >}})を参照してください。

SCIM API と対話するための実用的な Python の例は、[`wandb-scim`](https://github.com/wandb/examples/tree/master/wandb-scim) リポジトリをご覧ください。

### サポート機能
- フィルタリング: `/Users` と `/Groups` エンドポイントでのフィルタリングをサポート
- PATCH 操作: リソースの一部更新に PATCH をサポート
- ETag サポート: 競合検出のための ETag を用いた条件付き更新
- サービスアカウント認証: 組織のサービスアカウントが API に アクセス 可能

{{% alert %}}
複数の Enterprise の [Multi-tenant SaaS]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) 組織の管理者である場合、あなたの API キーで送信した SCIM API リクエストが正しい組織に影響するよう、SCIM API リクエストを送る組織を設定する必要があります。プロフィール画像をクリックし、**User Settings** をクリック、**Default API organization** を確認してください。

選択したホスティングオプションにより、このページの例で使用している `<host-url>` プレースホルダの 値 が決まります。

また、例では `abc` や `def` のようなユーザー ID を使用しています。実際のリクエスト／レスポンスではユーザー ID はハッシュ化された 値 になります。
{{% /alert %}}

## 認証

主な違いを確認したうえで、ユーザー本人のアイデンティティかサービスアカウントのどちらで認証するかを選択します。

### 主な違い
- 想定利用者: ユーザーは対話的で単発の管理作業に最適。サービスアカウントは自動化やインテグレーション（CI/CD、プロビジョニング ツール）に最適。
- 資格情報: ユーザーはユーザー名と APIキー を送信。サービスアカウントは APIキー のみ（ユーザー名なし）。
- Authorization ヘッダーのペイロード: ユーザーは `username:API-KEY` をエンコード。サービスアカウントは `:API-KEY`（先頭コロン）をエンコード。
- スコープと権限: どちらも管理者権限が必要。サービスアカウントは組織スコープでヘッドレスなため、自動化の監査証跡が明確。
- 資格情報の取得場所: ユーザーは User Settings から自分の APIキー をコピー。サービスアカウントのキーは組織の Service account タブにある。
- SaaS クラウドの組織宛先: 複数組織の管理者は、意図した組織にリクエストが作用するよう Default API organization を設定する。

### Users
対話的な管理タスクを行う際は、個人の管理者資格情報を使用します。HTTP の `Authorization` ヘッダーは `Basic <base64(username:API-KEY)>` の形式で構築します。

例: `demo:p@55w0rd` として認可する場合:
```bash
Authorization: Basic ZGVtbzpwQDU1dzByZA==
```

### サービスアカウント
自動化やインテグレーションには、組織スコープのサービスアカウントを使用します。HTTP の `Authorization` ヘッダーは `Basic <base64(:API-KEY)>` の形式で構築します（先頭のコロンと空のユーザー名に注意）。サービスアカウントの APIキー は組織のダッシュボードの **Service account** タブにあります。詳細は [Organization-scoped service accounts]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md/#organization-scoped-service-accounts" lang="ja" >}}) を参照してください。

例: APIキー `sa-p@55w0rd` で認可する場合:
```bash
Authorization: Basic OnNhLXBANTV3MHJk
```

## User 管理

SCIM の user リソースは W&B の User に対応します。これらのエンドポイントで組織内の User を管理します。

### Get user

組織内の特定の User の情報を取得します。

#### エンドポイント
- URL: `<host-url>/scim/Users/{id}`
- Method: GET

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| `id` | string | はい | User の一意な ID |

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
        "Value": "dev-user1@example.com",
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

### List users

組織内のすべての User を一覧取得します。

#### User のフィルタ

`/Users` エンドポイントは、ユーザー名またはメールでのフィルタリングをサポートします:

- `userName eq "value"` - ユーザー名でフィルタ
- `emails.value eq "value"` - メールアドレスでフィルタ

##### 例
```bash
GET /scim/Users?filter=userName eq "john.doe"
GET /scim/Users?filter=emails.value eq "john@example.com"
```

#### エンドポイント
- URL: `<host-url>/scim/Users`
- Method: GET

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
                "Value": "dev-user1@example.com",
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

### Create User

組織内に新しい User を作成します。

#### エンドポイント
- URL: `<host-url>/scim/Users`
- Method: POST

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| `emails` | array | はい | メール オブジェクトの配列。プライマリメールを含める必要あり |
| `userName` | string | はい | 新規 User のユーザー名 |

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
            "value": "dev-user2@example.com"
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
            "value": "dev-user2@example.com"
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
        "Value": "dev-user2@example.com",
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
        "Value": "dev-user2@example.com",
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

### Delete User

{{% alert color="warning" title="管理者 アクセス を維持する" %}}
常に少なくとも 1 名の管理者 User がインスタンスまたは組織内に存在するようにしてください。そうでない場合、誰も組織の W&B アカウントを 設定 または維持できなくなります。組織が SCIM やその他の自動プロセスで W&B からユーザーのプロビジョニング解除を行っていると、最後の管理者が誤ってインスタンスや組織から削除される可能性があります。

運用手順の策定支援、または管理者 アクセス の復旧が必要な場合は、[support](mailto:support@wandb.com) までご連絡ください。
{{% /alert %}}

組織から User を完全に削除します。

#### エンドポイント
- URL: `<host-url>/scim/Users/{id}`
- Method: DELETE

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| `id` | string | はい | 削除する User の一意な ID |

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
一時的に無効化したい場合は、`PATCH` エンドポイントを使う [Deactivate user](#deactivate-user) API を参照してください。
{{% /alert %}}

### User のメールを更新
User のプライマリメールアドレスを更新します。
Multi-tenant クラウド ではサポートされません。組織がユーザーのアカウントを管理しないためです。

#### エンドポイント
- URL: `<host-url>/scim/Users/{id}`
- Method: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| `id` | string | はい | User の一意な ID |
| `op` | string | はい | `replace` |
| `path` | string | はい | `emails` |
| `value` | array | はい | 新しいメールオブジェクトを含む配列 |

#### 例

{{< tabpane text=true >}}
{{% tab header="Update Email Request" %}}
```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "emails",
            "value": [
                {
                    "value": "newemail@example.com",
                    "primary": true
                }
            ]
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Update Email Response" %}}
```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "Dev User 1",
    "emails": {
        "Value": "newemail@example.com",
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

### User の表示名を更新

User の表示名を更新します。
Multi-tenant クラウド ではサポートされません。組織がユーザーのアカウントを管理しないためです。

#### エンドポイント
- URL: `<host-url>/scim/Users/{id}`
- Method: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| `id` | string | はい | User の一意な ID |
| `op` | string | はい | `replace` |
| `path` | string | はい | `displayName` |
| `value` | string | はい | 新しい表示名 |

#### 例

{{< tabpane text=true >}}
{{% tab header="Update Display Name Request" %}}
```bash
PATCH /scim/Users/abc
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "replace",
            "path": "displayName",
            "value": "John Doe"
        }
    ]
}
```
{{% /tab %}}
{{% tab header="Update Display Name Response" %}}
```bash
(Status 200)
```

```json
{
    "active": true,
    "displayName": "John Doe",
    "emails": {
        "Value": "dev-user1@example.com",
        "Display": "",
        "Type": "",
        "Primary": true
    },
    "id": "abc",
    "meta": {
        "resourceType": "User",
        "created": "2025-7-01T00:00:00Z",
        "lastModified": "2025-7-01T00:00:00Z",
        "location": "users/dev-user1"
    },
    "schemas": [
        "urn:ietf:params:scim:schemas:core:2.0:User"
    ],
    "userName": "dev-user1"
}
```
{{% /tab %}}
{{< /tabpane >}}

### Deactivate user

組織内の User を無効化します。

#### エンドポイント
- URL: `<host-url>/scim/Users/{id}`
- Method: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| `id` | string | はい | 無効化する User の一意な ID |
| `op` | string | はい | `replace` |
| `value` | object | はい | `{"active": false}` を含む オブジェクト |

{{% alert %}}
User の無効化と再有効化は、[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではサポートされません。
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
        "Value": "dev-user1@example.com",
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

### Reactivate User

以前に無効化した User を組織内で再有効化します。

#### エンドポイント
- URL: `<host-url>/scim/Users/{id}`
- Method: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| `id` | string | はい | 再有効化する User の一意な ID |
| `op` | string | はい | `replace` |
| `value` | object | はい | `{"active": true}` を含む オブジェクト |

{{% alert %}}
User の無効化と再有効化は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではサポートされません。
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
        "Value": "dev-user1@example.com",
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

### 組織ロールを割り当てる

User に組織レベルのロールを割り当てます。

#### エンドポイント
- URL: `<host-url>/scim/Users/{id}`
- Method: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| `id` | string | はい | User の一意な ID |
| `op` | string | はい | `replace` |
| `path` | string | はい | `organizationRole` |
| `value` | string | はい | ロール名（`admin` または `member`） |

{{% alert %}}
`viewer` ロールは廃止され、UI では設定できなくなりました。SCIM で `viewer` を割り当てようとすると、W&B はその User に `member` ロールを割り当てます。可能であれば User には自動的に Models と Weave のシートが割り当てられます。割り当てできない場合は `Seat limit reached` エラーが記録されます。**Registry** を使用している組織では、組織レベルで可視なレジストリにおいて、その User には自動的に `viewer` ロールが割り当てられます。
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
        "Value": "dev-user1@example.com",
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

### Team ロールを割り当てる

User にチームレベルのロールを割り当てます。

#### エンドポイント
- URL: `<host-url>/scim/Users/{id}`
- Method: PATCH

#### パラメータ
| パラメータ | 型 | 必須 | 説明 |
|-----------|------|----------|-------------|
| `id` | string | はい | User の一意な ID |
| `op` | string | はい | `replace` |
| `path` | string | はい | `teamRoles` |
| `value` | array | はい | `teamName` と `roleName` を持つ オブジェクト の配列 |

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
        "Value": "dev-user1@example.com",
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

IAM で SCIM のグループを作成すると、W&B の Team が作成されてマッピングされます。その他の SCIM グループ操作も Team に対して作用します。

### サービスアカウント

SCIM を使って W&B の Team が作成されると、組織レベルのすべてのサービスアカウントが自動的にその Team に追加され、サービスアカウントの Team リソースへの アクセス が維持されます。

### グループのフィルタリング

`/Groups` エンドポイントは、特定の Team を検索するためのフィルタリングをサポートします:

#### サポートされるフィルタ
- `displayName eq "value"` - Team の表示名でフィルタ

#### 例
```bash
GET /scim/Groups?filter=displayName eq "engineering-team"
```

### Team を取得

- エンドポイント: `<host-url>/scim/Groups/{id}`
- Method: GET
- 説明: Team の一意な ID を指定して Team 情報を取得します。
- リクエスト例:

```bash
GET /scim/Groups/ghi
```

- レスポンス例:

```bash
(Status 200)
```

```json
{
    "displayName": "acme-devs",
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

### Team を一覧

- エンドポイント: `<host-url>/scim/Groups`
- Method: GET
- 説明: Team の一覧を取得します。
- リクエスト例:

```bash
GET /scim/Groups
```

- レスポンス例:

```bash
(Status 200)
```

```json
{
    "Resources": [
        {
            "displayName": "acme-devs",
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

- エンドポイント: `<host-url>/scim/Groups`
- Method: POST
- 説明: 新しい Team リソースを作成します。
- サポートされるフィールド:

| フィールド | 型 | 必須 |
| --- | --- | --- |
| `displayName` | String | はい |
| `members` | Multi-Valued Array | はい（サブフィールド `value` は必須で、User ID に対応） |
- リクエスト例:

`wandb-support` という Team を作成し、そのメンバーとして `dev-user2` を追加します。

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

- レスポンス例:

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

- エンドポイント: `<host-url>/scim/Groups/{id}`
- Method: PATCH
- 説明: 既存 Team のメンバー一覧を更新します。
- サポートされる操作: メンバーの `add`、メンバーの `remove`、メンバー全体の `replace`

{{% alert %}}
remove 操作は RFC 7644 の SCIM プロトコル仕様に従います。特定のユーザーを削除するにはフィルタ構文 `members[value eq "{user_id}"]` を使用し、全ユーザーを削除するには `members` を使用します。

ユーザーの識別: メンバー操作における `{user_id}` は次のいずれかです:
- W&B のユーザー ID
- メールアドレス（例: "user@example.com"）
{{% /alert %}}

{{% alert color="info" %}}
リクエスト中の `{team_id}` は実際の Team ID、`{user_id}` は実際のユーザー ID かメールアドレスに置き換えてください。
{{% /alert %}}

### Team メンバーを置換

Team のメンバーを新しい一覧で全置換します。

- エンドポイント: `<host-url>/scim/Groups/{id}`
- Method: PUT
- 説明: Team のメンバー一覧全体を置換します。

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PUT /scim/Groups/{team_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
    "displayName": "acme-devs",
    "members": [
        {
            "value": "{user_id_1}"
        },
        {
            "value": "{user_id_2}"
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
    "displayName": "acme-devs",
    "id": "ghi",
    "members": [
        {
            "Value": "user_id_1",
            "Ref": "",
            "Type": "",
            "Display": "user1"
        },
        {
            "Value": "user_id_2",
            "Ref": "",
            "Type": "",
            "Display": "user2"
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

ユーザーを Team に追加する

`dev-user2` を `acme-devs` に追加:

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
    "displayName": "acme-devs",
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

特定のユーザーを Team から削除する

`acme-devs` から `dev-user2` を削除:

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
    "displayName": "acme-devs",
    "id": "ghi",
    "members": [
        {
            "Value": "abc",
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

Team から全ユーザーを削除する

`acme-devs` から全ユーザーを削除:

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
    "displayName": "acme-devs",
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

### Team を削除

- Team の削除は、Team に紐づく追加 データ があるため、現時点では SCIM API ではサポートされていません。完全削除を確認するため、アプリから削除してください。

## Role リソース

SCIM の role リソースは W&B のカスタムロールに対応します。先述のとおり、`/Roles` エンドポイントは公式の SCIM スキーマの一部ではありませんが、W&B 組織でカスタムロールを自動管理できるように W&B が `/Roles` エンドポイントを追加しています。

### カスタムロールを取得

- エンドポイント: `<host-url>/scim/Roles/{id}`
- Method: GET
- 説明: ロールの一意な ID を指定して、カスタムロールの情報を取得します。
- リクエスト例:

```bash
GET /scim/Roles/abc
```

- レスポンス例:

```bash
(Status 200)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 事前定義ロールを示します
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
            "isInherited": true // member の事前定義ロールから継承
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

### カスタムロールを一覧

- エンドポイント: `<host-url>/scim/Roles`
- Method: GET
- 説明: W&B 組織内のすべてのカスタムロールの情報を取得します。
- リクエスト例:

```bash
GET /scim/Roles
```

- レスポンス例:

```bash
(Status 200)
```

```json
{
   "Resources": [
        {
            "description": "A sample custom role for example",
            "id": "Um9sZTo3",
            "inheritedFrom": "member", // カスタムロールが継承する事前定義ロールを示します
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
                    "isInherited": true // viewer の事前定義ロールから継承
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
            "inheritedFrom": "viewer", // カスタムロールが継承する事前定義ロールを示します
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
                    "isInherited": true // viewer の事前定義ロールから継承
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

### カスタムロールを作成

- エンドポイント: `<host-url>/scim/Roles`
- Method: POST
- 説明: W&B 組織に新しいカスタムロールを作成します。
- サポートされるフィールド:

| フィールド | 型 | 説明 |
| --- | --- | --- |
| `name` | String | カスタムロールの名前 |
| `description` | String | カスタムロールの説明 |
| `permissions` | Object array | 権限オブジェクトの配列。各オブジェクトには `name` 文字列フィールドが含まれ、`w&bobject:operation` の形式の 値 を取ります。例えば、W&B の run の削除権限なら `name` は `run:delete`。 |
| `inheritedFrom` | String | カスタムロールが継承する事前定義ロール。`member` または `viewer` のいずれか。 |
- リクエスト例:

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

- レスポンス例:

```bash
(Status 201)
```

```json
{
    "description": "A sample custom role for example",
    "id": "Um9sZTo3",
    "inheritedFrom": "member", // 事前定義ロールを示します
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
            "isInherited": true // member の事前定義ロールから継承
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

### カスタムロールを更新

#### ロールに権限を追加

- エンドポイント: `<host-url>/scim/Roles/{id}`
- Method: PATCH
- 説明: 既存のカスタムロールに権限を追加します。

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Roles/{role_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "add",
            "path": "permissions",
            "value": [
                {
                    "name": "project:delete"
                },
                {
                    "name": "run:stop"
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

更新されたロールが返り、新しい権限が追加されています。
{{% /tab %}}
{{< /tabpane >}}

#### ロールから権限を削除

- エンドポイント: `<host-url>/scim/Roles/{id}`
- Method: PATCH
- 説明: 既存のカスタムロールから権限を削除します。

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PATCH /scim/Roles/{role_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
    "Operations": [
        {
            "op": "remove",
            "path": "permissions",
            "value": [
                {
                    "name": "project:update"
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

更新されたロールが返り、指定した権限が削除されています。
{{% /tab %}}
{{< /tabpane >}}

### カスタムロールを置換

- エンドポイント: `<host-url>/scim/Roles/{id}`
- Method: PUT
- 説明: カスタムロール定義全体を置換します。

{{< tabpane text=true >}}
{{% tab header="Request" %}}
```bash
PUT /scim/Roles/{role_id}
```

```json
{
    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
    "name": "Updated custom role",
    "description": "Updated description for the custom role",
    "permissions": [
        {
            "name": "project:read"
        },
        {
            "name": "run:read"
        },
        {
            "name": "artifact:read"
        }
    ],
    "inheritedFrom": "viewer"
}
```
{{% /tab %}}
{{% tab header="Response" %}}
```bash
(Status 200)
```

完全に置換されたロール定義が返ります。
{{% /tab %}}
{{< /tabpane >}}

### カスタムロールを削除

- エンドポイント: `<host-url>/scim/Roles/{id}`
- Method: DELETE
- 説明: W&B 組織のカスタムロールを削除します。注意して使用してください。カスタムロールが継承していた事前定義ロールが、そのロールに割り当てられていたすべての User に割り当て直されます。
- リクエスト例:

```bash
DELETE /scim/Roles/abc
```

## 高度な機能

### ETag サポート

SCIM API は、同時更新の競合を防ぐための条件付き更新に ETag をサポートします。ETag はレスポンスヘッダーの `ETag` と、`meta.version` フィールドに返されます。

#### ETag の使い方

ETag を使用するには:

1. 現在の ETag を取得: リソースを GET した際、レスポンスに含まれる ETag ヘッダーをメモ
2. 条件付き更新: 更新時に `If-Match` ヘッダーにその ETag を含める

#### 例

```
# ユーザーを取得して ETag を確認
GET /scim/Users/abc
# レスポンスに: ETag: W/"xyz123"

# ETag を付けて更新
PATCH /scim/Users/abc
If-Match: W/"xyz123"

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

`412 Precondition Failed` エラーは、取得後にリソースが変更されたことを示します。

### エラーハンドリング

SCIM API は標準的な SCIM エラーレスポンスを返します:

| ステータスコード | 説明 |
|-------------|-------------|
| `200` | Success |
| `201` | Created |
| `204` | No Content（削除成功） |
| `400` | Bad Request - パラメータまたはリクエストボディが不正 |
| `401` | Unauthorized - 認証失敗 |
| `403` | Forbidden - 権限不足 |
| `404` | Not Found - リソースが存在しない |
| `409` | Conflict - リソースがすでに存在 |
| `412` | Precondition Failed - ETag の不一致 |
| `500` | Internal Server Error |

### デプロイタイプごとの実装差異

W&B は 2 つの SCIM API 実装を維持しており、機能が異なります:

| 機能 | 専用クラウド | セルフマネージド |
|---------|-------------------|------------|
| User のメール更新 | - | &check; |
| User の表示名更新 | - | &check; |
| User の無効化／再有効化 | - | &check; |
| User あたり複数メール | &check; | - |

## 制限事項

- 最大取得件数: 1 リクエストあたり 9999 件
- シングルテナント 環境: User あたりメールは 1 件のみサポート
- Team の削除: SCIM では非対応（W&B の Web インターフェースを使用）
- User の無効化／再有効化: SaaS クラウド 環境では非対応
- シート上限: 組織のシート上限に達している場合、操作が失敗することがあります