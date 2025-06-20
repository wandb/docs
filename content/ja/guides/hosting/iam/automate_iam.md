---
title: ユーザーとチームの管理を自動化する
menu:
  default:
    identifier: ja-guides-hosting-iam-automate_iam
    parent: identity-and-access-management-iam
weight: 3
---

## SCIM API

SCIM API を使用して、ユーザーやその所属するチームを効率的かつ再現可能な方法で管理します。また、SCIM API を使用してカスタムロールを管理したり、 W&B 組織内のユーザーにロールを割り当てたりすることもできます。ロールエンドポイントは公式の SCIM スキーマの一部ではありません。 W&B はカスタムロールの自動管理をサポートするためにロールエンドポイントを追加しています。

SCIM API は特に以下の場合に役立ちます：

* 大規模なユーザーのプロビジョニングおよびプロビジョニング解除の管理
* SCIMサポートのアイデンティティプロバイダーを使用してユーザーを管理

SCIM API には大きく分けて 3 つのカテゴリがあります - **User** 、 **Group** 、および **Roles** 。

### User SCIM API

[User SCIM API]({{< relref path="./scim.md#user-resource" lang="ja" >}}) を使用すると、ユーザーの作成、無効化、詳細の取得、または W&B 組織内の全ユーザーの一覧を表示できます。この API は、組織内のユーザーに事前定義されたロールまたはカスタムロールを割り当てることもサポートしています。

{{% alert %}}
`DELETE User` エンドポイントを使用して、W&B 組織内のユーザーを無効化します。無効化されたユーザーはサインインできなくなります。ただし、無効化されたユーザーは引き続き組織のユーザーリストに表示されます。

無効化されたユーザーをユーザーリストから完全に削除するには、 [ユーザーを組織から削除]({{< relref path="access-management/manage-organization.md#remove-a-user" lang="ja" >}}) する必要があります。

必要に応じて無効化されたユーザーを再有効化することが可能です。
{{% /alert %}}

### Group SCIM API

[Group SCIM API]({{< relref path="./scim.md#group-resource" lang="ja" >}}) は、組織内での W&B チームの作成や削除を含めた管理を行うことができます。 `PATCH Group` を使用して、既存のチームにユーザーを追加または削除します。

{{% alert %}}
W&B には、 `グループ内のユーザーが同じロールを持つ` という概念はありません。 W&B チームはグループに非常に似ており、さまざまな役割を持った多様な個人が関連するプロジェクトで共同作業することができます。チームは異なるユーザーグループで構成されることができます。チーム内の各ユーザーに、チームの管理者、メンバー、閲覧者、またはカスタムロールのいずれかのロールを割り当てます。

グループと W&B チームの類似性のために、 W&B は Group SCIM API エンドポイントを W&B チームにマップしています。
{{% /alert %}}

### Custom role API

[Custom role SCIM API]({{< relref path="./scim.md#role-resource" lang="ja" >}}) は、組織内でのカスタムロールの作成、一覧表示、更新を含めた管理を行うことができます。

{{% alert color="secondary" %}}
カスタムロールを削除する際は注意してください。

`DELETE Role` エンドポイントを使用して W&B 組織内でカスタムロールを削除します。カスタムロールが継承する事前定義ロールは、操作が実行される前にカスタムロールに割り当てられたすべてのユーザーに割り当てられます。

`PUT Role` エンドポイントを使用してカスタムロールの継承ロールを更新します。この操作は、カスタムロールの既存の、つまり非継承のカスタム許可には影響しません。
{{% /alert %}}

## W&B Python SDK API

SCIM API がユーザーとチームの管理を自動化するのと同様に、一部のメソッドを使用して [W&B Python SDK API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) も同様の目的で使用できます。以下のメソッドに注意してください：

| メソッド名 | 目的 |
|-------------|---------|
| `create_user(email, admin=False)` | ユーザーを組織に追加し、オプションで組織の管理者に設定します。 |
| `user(userNameOrEmail)` | 組織に既に存在するユーザーを返します。 |
| `user.teams()` | ユーザーのチームを返します。ユーザーオブジェクトは user(userNameOrEmail) メソッドを使用して取得できます。 |
| `create_team(teamName, adminUserName)` | 新しいチームを作成し、オプションで組織レベルのユーザーをチーム管理者に設定します。 |
| `team(teamName)` | 組織に既に存在するチームを返します。 |
| `Team.invite(userNameOrEmail, admin=False)` | ユーザーをチームに追加します。team(teamName) メソッドを使用してチームオブジェクトを取得できます。 |
| `Team.create_service_account(description)` | サービスアカウントをチームに追加します。team(teamName) メソッドを使用してチームオブジェクトを取得できます。 |
| `Member.delete()` | チームからメンバーを削除します。team オブジェクトの `members` 属性を使用してチーム内のメンバーオブジェクトのリストを取得できます。そして、 team(teamName) メソッドを使用してチームオブジェクトを取得できます。 |