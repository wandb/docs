---
title: ユーザーおよびチーム管理を自動化する
menu:
  default:
    identifier: ja-guides-hosting-iam-automate_iam
    parent: identity-and-access-management-iam
weight: 3
---

## SCIM API

SCIM API を使って、ユーザーやその所属するチームを効率的かつ反復可能な方法で管理できます。また、SCIM API を利用してカスタムロールの管理や W&B 組織内のユーザーへのロール割り当ても行えます。ロール関連のエンドポイントは公式な SCIM スキーマには含まれていませんが、W&B ではカスタムロール管理の自動化をサポートするために独自に追加しています。

SCIM API は、特に次のような場合に便利です。

* 大規模なユーザーの追加や削除を効率化したいとき
* SCIM をサポートする IdP（アイデンティティプロバイダー）でユーザー管理したいとき

SCIM API には、大きく分けて **User**、**Group**、**Roles** の 3 カテゴリがあります。

### User SCIM API

[User SCIM API]({{< relref path="./scim.md#user-resource" lang="ja" >}}) では、ユーザーの作成、無効化、詳細取得、組織内の全ユーザー一覧取得ができます。また、組織内のユーザーにあらかじめ設定されたロールやカスタムロールを割り当てることも可能です。

{{% alert %}}
`DELETE User` エンドポイントで、W&B 組織のユーザーを無効化できます。無効化されたユーザーはサインインできませんが、組織のユーザー一覧には残ります。

無効化されたユーザーをユーザー一覧から完全に削除するには、[ユーザーを組織から削除]({{< relref path="access-management/manage-organization.md#remove-a-user" lang="ja" >}})する必要があります。

必要に応じて、無効化したユーザーを再度有効化することもできます。
{{% /alert %}}

### Group SCIM API

[Group SCIM API]({{< relref path="./scim.md#group-resource" lang="ja" >}}) を使うと、W&B チームの管理（チームの作成や削除など）が可能です。既存チームにユーザーを追加・削除するには `PATCH Group` を利用してください。

{{% alert %}}
W&B では「同じロールを持つユーザーグループ」という概念はありません。W&B のチームはグループに近く、異なるロールを持つ様々なペルソナが、関連するプロジェクト群で共同作業できるようになっています。チームはさまざまなユーザーグループで構成できます。チーム内の各ユーザーには、チーム管理者、メンバー、ビューア、またはカスタムロールを割り当てます。

Group SCIM API のエンドポイントは、グループと W&B チームの類似性から W&B チームにマッピングされています。
{{% /alert %}}

### Custom role API

[Custom role SCIM API]({{< relref path="./scim.md#role-resource" lang="ja" >}}) を使うと、カスタムロールの管理（作成、一覧取得、更新など）が行えます。

{{% alert color="secondary" %}}
カスタムロールの削除は慎重に行ってください。

W&B 組織内でカスタムロールを削除する場合は `DELETE Role` エンドポイントを使用します。削除操作の前に、そのカスタムロールが割り当てられているすべてのユーザーには、カスタムロールが継承している定義済みロールが割り当てられます。

`PUT Role` エンドポイントでカスタムロールの継承元ロールを更新できます。この操作は、カスタムロール内の既存（つまり継承していない）カスタム権限には影響しません。
{{% /alert %}}

## W&B Python SDK API

SCIM API を使ってユーザーやチームの管理を自動化できるのと同じように、[W&B Python SDK API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) に用意されているいくつかのメソッドを使って同様の管理が可能です。主なメソッドは以下の通りです。

| メソッド名 | 目的 |
|-------------|---------|
| `create_user(email, admin=False)` | ユーザーを組織に追加し、必要に応じて組織管理者に設定します。|
| `user(userNameOrEmail)` | 組織内の既存ユーザーを取得します。|
| `user.teams()` | ユーザーが所属するチームを取得します。ユーザーオブジェクトは user(userNameOrEmail) メソッドで取得できます。|
| `create_team(teamName, adminUserName)` | 新しいチームを作成し、組織レベルのユーザーをチーム管理者に設定（任意）します。|
| `team(teamName)` | 組織内の既存チームを取得します。|
| `Team.invite(userNameOrEmail, admin=False)` | チームにユーザーを追加します。team(teamName) メソッドでチームオブジェクトを取得できます。|
| `Team.create_service_account(description)` | サービスアカウントをチームに追加します。team(teamName) メソッドでチームオブジェクトを取得できます。|
| `Member.delete()` | チームからメンバーユーザーを削除します。チームオブジェクトの `members` 属性でチーム内のメンバーオブジェクト一覧が取得でき、team(teamName) メソッドでチームオブジェクトが取得できます。|