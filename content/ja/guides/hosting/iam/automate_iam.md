---
title: ユーザーとチームの管理を自動化
menu:
  default:
    identifier: ja-guides-hosting-iam-automate_iam
    parent: identity-and-access-management-iam
weight: 3
---

## SCIM API

SCIM API を使うと、Users と彼らが所属する Teams を効率的かつ再現可能な方法で管理できます。また、SCIM API を使用して、カスタムロールの管理や、W&B の組織内の Users へのロール割り当ても行えます。ロール用エンドポイントは公式の SCIM スキーマの一部ではありません。W&B では、カスタムロールの自動管理をサポートするためにロール用エンドポイントを追加しています。

SCIM API は特に次のような場合に有用です:

* 大規模な User のプロビジョニングとデプロビジョニングを管理する
* SCIM に対応したアイデンティティ プロバイダで Users を管理する

SCIM API は大きく **User**、**Group**、**Roles** の 3 つに分類されます。

### User SCIM API

[User SCIM API]({{< relref path="./scim.md#user-resource" lang="ja" >}}) では、W&B の組織内で User を作成・無効化・詳細取得したり、すべての Users を一覧表示できます。この API は、組織内の Users に対してあらかじめ定義されたロールやカスタムロールを割り当てることもサポートします。

{{% alert %}}
`DELETE User` エンドポイントで、W&B の組織内の User を無効化できます。無効化された Users はサインインできなくなります。ただし、無効化された Users は組織の User リストには引き続き表示されます。

無効化した User を User リストから完全に削除するには、[組織からその user を削除]({{< relref path="access-management/manage-organization.md#remove-a-user" lang="ja" >}}) する必要があります。

必要に応じて、無効化した User を再有効化することも可能です。
{{% /alert %}}

### Group SCIM API

[Group SCIM API]({{< relref path="./scim.md#group-resource" lang="ja" >}}) では、W&B Teams の管理（組織内の Teams の作成や削除など）が行えます。既存の Team に Users を追加または削除するには `PATCH Group` を使用します。

{{% alert %}}
W&B に `group of users having the same role` という概念はありません。W&B の Team は group に近く、異なるロールを持つ多様なペルソナが、関連する一連の Projects に共同で取り組めます。Teams は異なる Users のグループで構成され得ます。Team 内の各 User に、team admin、member、viewer、またはカスタムロールのいずれかのロールを割り当ててください。

グループと W&B の Teams が類似しているため、W&B は Group SCIM API のエンドポイントを W&B の Teams にマッピングしています。
{{% /alert %}}

### カスタムロール API

[Custom role SCIM API]({{< relref path="./scim.md#role-resource" lang="ja" >}}) では、組織内でカスタムロールを作成・一覧表示・更新するなど、カスタムロールの管理が行えます。

{{% alert color="secondary" %}}
カスタムロールの削除は慎重に行ってください。

`DELETE Role` エンドポイントで、W&B の組織内のカスタムロールを削除できます。この操作の前に、カスタムロールを付与されているすべての Users には、そのカスタムロールが継承している事前定義ロールが割り当てられます。

`PUT Role` エンドポイントで、カスタムロールが継承しているロールを更新できます。この操作は、既存の（つまり継承ではない）カスタム権限には影響しません。
{{% /alert %}}

## W&B Python SDK API

SCIM API で Users や Teams の管理を自動化できるのと同様に、その目的には [W&B Python SDK API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) に用意されているいくつかのメソッドも利用できます。次のメソッドを覚えておくと便利です:

| メソッド名 | 目的 |
|-------------|---------|
| `create_user(email, admin=False)` | 組織に User を追加し、必要に応じて組織の admin に指定します。 |
| `user(userNameOrEmail)` | 組織内に存在する User を返します。 |
| `user.teams()` | その User が所属する Teams を返します。User のオブジェクトは user(userNameOrEmail) メソッドで取得できます。 |
| `create_team(teamName, adminUserName)` | 新しい Team を作成し、必要に応じて組織レベルの User を Team admin に指定します。 |
| `team(teamName)` | 組織内に存在する Team を返します。 |
| `Team.invite(userNameOrEmail, admin=False)` | Team に User を追加します。Team のオブジェクトは team(teamName) メソッドで取得できます。 |
| `Team.create_service_account(description)` | Team にサービスアカウントを追加します。Team のオブジェクトは team(teamName) メソッドで取得できます。 |
|` Member.delete()` | Team からメンバー User を削除します。Team のオブジェクトの `members` 属性を使うと、Team 内のメンバーオブジェクトの一覧を取得できます。なお、Team のオブジェクトは team(teamName) メソッドで取得できます。 |