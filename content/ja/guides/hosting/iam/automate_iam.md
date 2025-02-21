---
title: Automate user and team management
menu:
  default:
    identifier: ja-guides-hosting-iam-automate_iam
    parent: identity-and-access-management-iam
weight: 3
---

## SCIM API

SCIM API を使用すると、 ユーザー 、および ユーザー が所属する Teams を効率的かつ反復可能な方法で管理できます。 SCIM API を使用して、カスタムロールの管理や、W&B Organization 内の ユーザー へのロールの割り当てもできます。ロールエンドポイントは、公式の SCIM スキーマの一部ではありません。W&B は、カスタムロールの自動管理をサポートするためにロールエンドポイントを追加します。

SCIM API は、特に次のような場合に役立ちます。

*   大規模な ユーザー のプロビジョニングとプロビジョニング解除を管理する
*   SCIM をサポートする ID プロバイダーで ユーザー を管理する

SCIM API には、大きく分けて **User** 、 **Group** 、 **Roles** の 3 つのカテゴリーがあります。

### User SCIM API

[User SCIM API]({{< relref path="./scim.md#user-resource" lang="ja" >}}) を使用すると、W&B Organization 内の ユーザー の作成、非アクティブ化、詳細の取得、またはすべての ユーザー の一覧表示ができます。この API は、定義済みのロールまたはカスタムロールを Organization 内の ユーザー に割り当てることもサポートしています。

{{% alert %}}
`DELETE User` エンドポイントを使用して、W&B Organization 内の ユーザー を非アクティブ化します。非アクティブ化された ユーザー はサインインできなくなります。ただし、非アクティブ化された ユーザー は、Organization の ユーザー リストに引き続き表示されます。

非アクティブ化された ユーザー を ユーザー リストから完全に削除するには、[Organization から ユーザー を削除する]({{< relref path="access-management/manage-organization.md#remove-a-user" lang="ja" >}})必要があります。

必要に応じて、非アクティブ化された ユーザー を再度有効にすることができます。
{{% /alert %}}

### Group SCIM API

[Group SCIM API]({{< relref path="./scim.md#group-resource" lang="ja" >}}) を使用すると、Organization 内の Teams の作成または削除など、W&B の Teams を管理できます。既存の Team に ユーザー を追加または削除するには、`PATCH Group` を使用します。

{{% alert %}}
W&B には、`同じロールを持つ ユーザー のグループ` という概念はありません。W&B の Team はグループによく似ており、異なるロールを持つ多様なペルソナが、関連する Projects のセットで共同作業できます。Teams は、異なる ユーザー のグループで構成できます。Team 内の各 ユーザー に、Team 管理者、メンバー、閲覧者、またはカスタムロールを割り当てます。

W&B は、グループと W&B の Teams の類似性から、Group SCIM API エンドポイントを W&B の Teams にマッピングします。
{{% /alert %}}

### Custom role API

[Custom role SCIM API]({{< relref path="./scim.md#role-resource" lang="ja" >}}) を使用すると、Organization 内のカスタムロールの作成、一覧表示、または更新など、カスタムロールを管理できます。

{{% alert color="secondary" %}}
カスタムロールを削除する際は注意してください。

`DELETE Role` エンドポイントを使用して、W&B Organization 内のカスタムロールを削除します。カスタムロールが継承する定義済みのロールは、操作前にカスタムロールが割り当てられているすべての ユーザー に割り当てられます。

`PUT Role` エンドポイントを使用して、カスタムロールの継承されたロールを更新します。この操作は、カスタムロールの既存の、つまり継承されていないカスタム権限には影響しません。
{{% /alert %}}

## W&B Python SDK API

SCIM API で ユーザー と Team の管理を自動化できるのと同じように、[W&B Python SDK API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) で利用可能な メソッド の一部もその目的で使用できます。次の メソッド に注意してください。

| Method name | Purpose |
|-------------|---------|
| `create_user(email, admin=False)` | ユーザー を Organization に追加し、オプションで Organization 管理者にします。 |
| `user(userNameOrEmail)` | Organization 内の既存の ユーザー を返します。 |
| `user.teams()` | ユーザー の Teams を返します。user(userNameOrEmail) メソッド を使用して ユーザー オブジェクトを取得できます。 |
| `create_team(teamName, adminUserName)` | 新しい Team を作成し、オプションで Organization レベルの ユーザー を Team 管理者にします。 |
| `team(teamName)` | Organization 内の既存の Team を返します。 |
| `Team.invite(userNameOrEmail, admin=False)` | ユーザー を Team に追加します。team(teamName) メソッド を使用して Team オブジェクトを取得できます。 |
| `Team.create_service_account(description)` | サービス アカウントを Team に追加します。team(teamName) メソッド を使用して Team オブジェクトを取得できます。 |
|` Member.delete()` | メンバー ユーザー を Team から削除します。team オブジェクトの `members` 属性を使用して、Team 内のメンバー オブジェクトのリストを取得できます。また、team(teamName) メソッド を使用して Team オブジェクトを取得できます。 |
