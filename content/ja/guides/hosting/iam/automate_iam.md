---
title: Automate user and team management
menu:
  default:
    identifier: ja-guides-hosting-iam-automate_iam
    parent: identity-and-access-management-iam
weight: 3
---

## SCIM API

SCIM APIを使用して、ユーザーとその所属するチームを効率的かつ繰り返し管理できます。また、SCIM APIはカスタム ロールを管理したり、W&B組織内のユーザーにロールを割り当てることもできます。ロールのエンドポイントは公式なSCIMスキーマの一部ではありません。W&Bはカスタム ロールの自動管理をサポートするためにロール エンドポイントを追加しています。

SCIM APIは次の場合に特に役立ちます：

* ユーザーのプロビジョニングとプロビジョニング解除を規模に応じて管理
* SCIMをサポートするアイデンティティ プロバイダーでユーザーを管理

SCIM APIは大きく分けて、**User**、**Group**、**Roles**の3つのカテゴリーに分かれます。

### User SCIM API

[User SCIM API]({{< relref path="./scim.md#user-resource" lang="ja" >}})を使用すると、W&B組織内でユーザーを作成したり、非アクティブ化したり、ユーザーの詳細を取得したり、すべてのユーザーを一覧表示したりできます。このAPIは、組織内のユーザーに事前定義されたロールやカスタム ロールを割り当てることもサポートしています。

{{% alert %}}
`DELETE User`エンドポイントを使用して、W&B組織内のユーザーを非アクティブ化します。非アクティブ化されたユーザーは、サインインできなくなります。しかし、非アクティブ化されたユーザーは引き続き組織のユーザーリストに表示されます。

非アクティブ化されたユーザーをユーザーリストから完全に削除するには、[組織からユーザーを削除する]({{< relref path="access-management/manage-organization.md#remove-a-user" lang="ja" >}})必要があります。

必要に応じて、非アクティブ化されたユーザーを再度有効にすることが可能です。
{{% /alert %}}

### Group SCIM API

[Group SCIM API]({{< relref path="./scim.md#group-resource" lang="ja" >}})を使用すると、W&Bチームを管理できます。これには、組織内のチームの作成や削除が含まれます。既存のチーム内でユーザーを追加または削除するには、`PATCH Group`を使用します。

{{% alert %}}
W&B内では`同じロールを持つユーザーのグループ`という概念はありません。W&Bのチームはグループに非常に似ており、さまざまなロールを持つ多様な人物が関連するプロジェクトのセットで協力して作業できます。チームはさまざまなユーザーグループで構成されることができます。各ユーザーに、チームの管理者、メンバー、ビューアー、またはカスタム ロールのいずれかを割り当てます。

W&BはグループとW&Bチームの類似性から、Group SCIM APIエンドポイントをW&Bチームにマッピングしています。
{{% /alert %}}

### Custom role API

[Custom role SCIM API]({{< relref path="./scim.md#role-resource" lang="ja" >}})を使用すると、カスタム ロールを管理できます。これには、組織内のカスタム ロールの作成、一覧表示、更新が含まれます。

{{% alert color="secondary" %}}
カスタム ロールを削除する際は注意が必要です。

`DELETE Role`エンドポイントを使用してW&B組織内のカスタム ロールを削除します。操作の前にカスタム ロールが割り当てられたすべてのユーザーには、カスタム ロールが継承している事前定義されたロールが割り当てられます。

`PUT Role`エンドポイントを使用して、カスタム ロールの継承ロールを更新します。この操作はカスタム ロール内の既存の、つまり非継承のカスタム権限には影響を与えません。
{{% /alert %}}

## W&B Python SDK API

SCIM APIがユーザーとチームの管理を自動化するのと同様に、[W&B Python SDK API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})で利用可能なメソッドをいくつか使用して、それを目的にすることもできます。次のメソッドに注意してください：

| メソッド名 | 目的 |
|-------------|---------|
| `create_user(email, admin=False)` | 組織にユーザーを追加し、オプションで組織管理者にすることができます。 |
| `user(userNameOrEmail)` | 組織内の既存のユーザーを返します。 |
| `user.teams()` | ユーザーのチームを返します。ユーザー オブジェクトは user(userNameOrEmail) メソッドを使用して取得できます。 |
| `create_team(teamName, adminUserName)` | 新しいチームを作成し、オプションで組織レベルのユーザーをチーム管理者にします。 |
| `team(teamName)` | 組織内の既存のチームを返します。 |
| `Team.invite(userNameOrEmail, admin=False)` | ユーザーをチームに追加します。チーム オブジェクトは team(teamName) メソッドを使用して取得できます。 |
| `Team.create_service_account(description)` | サービス アカウントをチームに追加します。チーム オブジェクトは team(teamName) メソッドを使用して取得できます。 |
|` Member.delete()` | メンバーのユーザーをチームから削除します。メンバー オブジェクトのリストは、チーム オブジェクトの `members` 属性を使用して取得できます。チーム オブジェクトは team(teamName) メソッドを使用して取得できます。 |