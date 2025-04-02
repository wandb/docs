---
title: Automate user and team management
menu:
  default:
    identifier: ja-guides-hosting-iam-automate_iam
    parent: identity-and-access-management-iam
weight: 3
---

## SCIM API

SCIM API を使用して、ユーザーと、ユーザーが所属する Teams を効率的かつ反復可能な方法で管理します。SCIM API を使用して、カスタムロールを管理したり、W&B organization 内の Users にロールを割り当てることもできます。ロールエンドポイントは、公式の SCIM スキーマの一部ではありません。W&B は、カスタムロールの自動管理をサポートするために、ロールエンドポイントを追加します。

SCIM API は、特に次のような場合に役立ちます。

* 大規模なユーザーのプロビジョニングとプロビジョニング解除を管理する
* SCIM をサポートする Identity Provider で Users を管理する

SCIM API には、大きく分けて **User**、**Group**、**Roles** の 3 つのカテゴリがあります。

### User SCIM API

[User SCIM API]({{< relref path="./scim.md#user-resource" lang="ja" >}}) を使用すると、User の作成、非アクティブ化、詳細の取得、または W&B organization 内のすべての Users の一覧表示が可能です。この API は、定義済みまたはカスタムロールを organization 内の Users に割り当てることもサポートしています。

{{% alert %}}
`DELETE User` エンドポイントを使用して、W&B organization 内の User を非アクティブ化します。非アクティブ化された Users は、サインインできなくなります。ただし、非アクティブ化された Users は、organization の User リストに引き続き表示されます。

非アクティブ化された User を User リストから完全に削除するには、[organization から User を削除する]({{< relref path="access-management/manage-organization.md#remove-a-user" lang="ja" >}})必要があります。

必要に応じて、非アクティブ化された User を再度有効にすることができます。
{{% /alert %}}

### Group SCIM API

[Group SCIM API]({{< relref path="./scim.md#group-resource" lang="ja" >}}) を使用すると、organization 内の Teams の作成や削除など、W&B Teams を管理できます。`PATCH Group` を使用して、既存の Team に Users を追加または削除します。

{{% alert %}}
W&B には、`同じロールを持つ Users のグループ` という概念はありません。W&B の Team はグループによく似ており、異なるロールを持つ多様なペルソナが、関連する Projects のセットで共同作業を行うことができます。Teams は、異なるグループの Users で構成できます。Team の各 User に、Team 管理者、メンバー、閲覧者、またはカスタムロールを割り当てます。

W&B は、グループと W&B Teams の類似性から、Group SCIM API エンドポイントを W&B Teams にマッピングします。
{{% /alert %}}

### Custom role API

[Custom role SCIM API]({{< relref path="./scim.md#role-resource" lang="ja" >}}) を使用すると、organization 内のカスタムロールの作成、一覧表示、または更新など、カスタムロールを管理できます。

{{% alert color="secondary" %}}
カスタムロールを削除する場合は注意してください。

`DELETE Role` エンドポイントを使用して、W&B organization 内のカスタムロールを削除します。カスタムロールが継承する定義済みのロールは、操作前にカスタムロールが割り当てられているすべての Users に割り当てられます。

`PUT Role` エンドポイントを使用して、カスタムロールの継承されたロールを更新します。この操作は、カスタムロール内の既存の、つまり継承されていないカスタム権限には影響しません。
{{% /alert %}}

## W&B Python SDK API

SCIM API で User と Team の管理を自動化できるのと同じように、[W&B Python SDK API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) で利用できる メソッド の一部もその目的に使用できます。次の メソッド に注意してください。

| Method name | Purpose |
|-------------|---------|
| `create_user(email, admin=False)` | organization に User を追加し、オプションで organization 管理者にします。 |
| `user(userNameOrEmail)` | organization 内の既存の User を返します。 |
| `user.teams()` | User の Teams を返します。user(userNameOrEmail) メソッド を使用して User オブジェクトを取得できます。 |
| `create_team(teamName, adminUserName)` | 新しい Team を作成し、オプションで organization レベルの User を Team 管理者にします。 |
| `team(teamName)` | organization 内の既存の Team を返します。 |
| `Team.invite(userNameOrEmail, admin=False)` | Team に User を追加します。team(teamName) メソッド を使用して Team オブジェクトを取得できます。 |
| `Team.create_service_account(description)` | Team にサービスアカウントを追加します。team(teamName) メソッド を使用して Team オブジェクトを取得できます。 |
|` Member.delete()` | Team からメンバー User を削除します。team オブジェクトの `members` 属性を使用して、Team 内のメンバー オブジェクトのリストを取得できます。また、team(teamName) メソッド を使用して Team オブジェクトを取得できます。 |
