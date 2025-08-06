---
title: ユーザーと Team 管理を自動化する
menu:
  default:
    identifier: automate_iam
    parent: identity-and-access-management-iam
weight: 3
---

## SCIM API

SCIM API を使うと、ユーザーやその所属 Teams を効率的かつ再現性のある方法で管理できます。また、SCIM API を使ってカスタムロールの管理や、W&B 組織内のユーザーへのロール割り当ても可能です。ロール用エンドポイントは公式の SCIM スキーマには含まれていませんが、W&B ではカスタムロールの自動管理をサポートするためにロール用エンドポイントを追加しています。

SCIM API は特に以下のような場合に便利です：

* 大規模なユーザーのプロビジョニングや削除を管理したい場合
* SCIM に対応した IdP（アイデンティティプロバイダー）でユーザー管理をしたい場合

SCIM API には大きく分けて **User**、**Group**、**Roles** の3種類があります。

### User SCIM API

[User SCIM API]({{< relref "./scim.md#user-resource" >}}) では、ユーザーの新規作成・無効化・詳細取得・組織内のすべてのユーザーのリスト取得が可能です。また、組織内のユーザーにあらかじめ定義されたロールやカスタムロールを割り当てることもできます。

{{% alert %}}
`DELETE User` エンドポイントを使って、W&B 組織内のユーザーを無効化できます。無効化されたユーザーはサインインできなくなりますが、組織のユーザー一覧には引き続き表示されます。

無効化されたユーザーをユーザーリストから完全に削除するには、[組織からユーザーを削除する]({{< relref "access-management/manage-organization.md#remove-a-user" >}}) 必要があります。

必要に応じて、無効化したユーザーを再有効化することも可能です。
{{% /alert %}}

### Group SCIM API

[Group SCIM API]({{< relref "./scim.md#group-resource" >}}) では、W&B Teams の管理（組織内でのチーム作成や削除など）が可能です。既存のチームにユーザーを追加・削除したい場合は、`PATCH Group` を利用します。

{{% alert %}}
W&B には「同じロールを持つユーザーのグループ」は存在しません。W&B における Team はグループに近い性質を持ち、さまざまなロールを持つ人が、関連する複数の Projects に共同で取り組むためのものです。Teams はユーザーごとに異なるグループで構成できます。それぞれの Team 内でユーザーごとに 「team admin」「member」「viewer」やカスタムロールなどの役割を割り当てて運用します。

W&B では groups と Teams の類似性から、Group SCIM API のエンドポイントを Teams として扱っています。
{{% /alert %}}

### Custom role API

[Custom role SCIM API]({{< relref "./scim.md#role-resource" >}}) を利用すると、カスタムロールの作成・一覧取得・更新など、組織内のカスタムロール管理が可能です。

{{% alert color="secondary" %}}
カスタムロールの削除は慎重に実行してください。

`DELETE Role` エンドポイントで W&B 組織内のカスタムロールを削除できます。カスタムロールが継承している事前定義ロールが、カスタムロールを割り当てられていた全ユーザーに自動的に再割り当てされます。

カスタムロールの継承元ロールを更新するには `PUT Role` エンドポイントを利用します。この操作は、すでに追加されている独自権限（つまり、継承していない部分）には影響しません。
{{% /alert %}}

## W&B Python SDK API

SCIM API でユーザーや Team の管理を自動化できるのと同様に、[W&B Python SDK API]({{< relref "/ref/python/public-api/api.md" >}}) にも一部その目的で使えるメソッドがあります。下記のメソッドを覚えておいてください：

| メソッド名 | 説明 |
|-------------|---------|
| `create_user(email, admin=False)` | ユーザーを組織に追加し、必要に応じて組織管理者権限も与えることができます。 |
| `user(userNameOrEmail)` | 組織内の既存ユーザーを返します。 |
| `user.teams()` | 該当ユーザーの Teams を返します。ユーザーオブジェクトは user(userNameOrEmail) メソッドで取得可能です。 |
| `create_team(teamName, adminUserName)` | 新しい Team を作成し、必要に応じて組織ユーザーを Team 管理者にできます。 |
| `team(teamName)` | 組織内の既存 Team を返します。 |
| `Team.invite(userNameOrEmail, admin=False)` | Team にユーザーを追加します。team(teamName) メソッドで Team オブジェクトを取得してください。 |
| `Team.create_service_account(description)` | Team にサービスアカウントを追加します。team(teamName) メソッドで Team オブジェクトを取得してください。 |
|` Member.delete()` | Team から member ユーザーを削除します。team オブジェクトの `members` 属性で members オブジェクトのリストを取得可能で、team(teamName) メソッドで team オブジェクトを取得できます。 |