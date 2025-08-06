---
title: アクセス管理
menu:
  default:
    identifier: access-management-intro
    parent: identity-and-access-management-iam
url: guides/hosting/iam/access-management-intro
cascade:
- url: guides/hosting/iam/access-management/:filename
weight: 2
---

## 組織内のユーザーおよびチームの管理

W&B への最初のサインアップユーザーで、その組織ドメインがユニークな場合、そのユーザーに *インスタンス管理者ロール* が割り当てられます。組織管理者が特定のユーザーにチーム管理者ロールを割り当てます。

{{% alert %}}
W&B では、組織内に 2 人以上のインスタンス管理者がいることを推奨しています。主な管理者が不在の場合でも管理業務が継続できるようにするベストプラクティスです。
{{% /alert %}}

*チーム管理者* とは、チーム内で管理権限を持つ組織内のユーザーです。

組織管理者は `https://wandb.ai/account-settings/` の組織アカウント設定にアクセスし、ユーザーの招待、ユーザーのロールの割り当て・更新、チームの作成、組織からのユーザー削除、課金管理者の割り当てなどが行えます。詳細は [ユーザーの追加と管理]({{< relref "./manage-organization.md#add-and-manage-users" >}}) をご覧ください。

組織管理者がチームを作成すると、インスタンス管理者またはチーム管理者は以下の操作ができます:

- デフォルトでは、チームへのユーザー招待や削除は管理者のみが可能です。この振る舞いを変更したい場合は [Team settings]({{< relref "/guides/models/app/settings-page/team-settings.md#privacy" >}}) をご参照ください。
- チームメンバーのロールを割り当てたり更新したりできます。
- 新たに組織に参加したユーザーを自動的にチームに追加できます。

組織管理者とチーム管理者はともに `https://wandb.ai/<your-team-name>` のチームダッシュボードを使い、チームの管理を行います。詳細やチームのデフォルト公開範囲設定については [チームの追加と管理]({{< relref "./manage-organization.md#add-and-manage-teams" >}}) をご確認ください。

## 管理者アクセスの維持

常にインスタンスや組織内に最低 1 人の管理者ユーザーが存在していることを確認してください。万が一管理者がいなくなると、組織の W&B アカウントの設定・管理ができなくなります。

ユーザーが手動で管理されている場合、管理者アクセス権限があればユーザーの削除が可能です（他の管理者も含む）。この制限により、唯一の管理者ユーザーが誤って削除されるリスクを低減します。

一方、自動化プロセスでユーザーの削除を行っている場合、誤って最後の管理者が削除される可能性があります。

運用手順の策定や管理者アクセスの復旧などでお困りの際は、[サポート](mailto:support@wandb.com)までご連絡ください。

## 特定の Projects への公開範囲を制限する

W&B Project のスコープを定義して、誰がその Project を閲覧・編集したり、W&B Runs を送信できるかを制限できます。特に、機密性の高いデータを扱うチームの場合 Project の公開範囲制限が有用です。

Project の公開範囲は、組織管理者、チーム管理者、または Project のオーナーが設定および編集できます。

詳しくは [Project visibility]({{< relref "./restricted-projects.md" >}}) をご覧ください。