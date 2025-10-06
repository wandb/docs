---
title: アクセス管理
cascade:
- url: /ja/guides/hosting/iam/access-management/:filename
menu:
  default:
    identifier: ja-guides-hosting-iam-access-management-_index
    parent: identity-and-access-management-iam
url: /ja/guides/hosting/iam/access-management-intro
weight: 2
---

## 組織内でのユーザーとチームの管理

ユニークな組織ドメインで W&B に初めてサインアップしたユーザーは、その組織の*インスタンス管理者ロール*に割り当てられます。組織管理者は特定のユーザーにチーム管理者ロールを割り当てます。

{{% alert %}}
W&B は、組織に複数のインスタンス管理者を持つことを推奨しています。これは、主な管理者が不在の場合にも管理業務を継続できるようにするためのベストプラクティスです。
{{% /alert %}}

*チーム管理者*は、チーム内で管理権限を持つ組織内のユーザーです。

組織管理者は、`https://wandb.ai/account-settings/` の組織アカウント設定にアクセスして、ユーザーを招待したり、ユーザーの役割を割り当てたり更新したり、チームを作成したり、組織からユーザーを削除したり、請求管理者を割り当てたりすることができます。詳細については、[ユーザーの追加と管理]({{< relref path="./manage-organization.md#add-and-manage-users" lang="ja" >}})を参照してください。

組織管理者がチームを作成すると、インスタンス管理者またはチーム管理者は次のことができます：

- デフォルトでは、管理者のみがそのチームにユーザーを招待したり、チームからユーザーを削除したりできます。この振る舞いを変更するには、[チーム設定]({{< relref path="/guides/models/app/settings-page/teams.md#privacy" lang="ja" >}})を参照してください。
- チームメンバーの役割を割り当てたり更新したりします。
- 組織に参加した際に自動的に新しいユーザーをチームに追加します。

組織管理者とチーム管理者は、`https://wandb.ai/<your-team-name>` のチームダッシュボードを使用してチームを管理します。詳細とチームのデフォルトの公開範囲を設定するには、[チームの追加と管理]({{< relref path="./manage-organization.md#add-and-manage-teams" lang="ja" >}})を参照してください。

## 特定のプロジェクトへの公開範囲の制限

W&B プロジェクトの範囲を定義して、誰がそのプロジェクトを閲覧、編集、そして W&B の run をサブミットできるかを制限します。プロジェクトを閲覧できる人を制限することは、特にチームが機密または秘密のデータを扱う場合に役立ちます。

組織管理者、チーム管理者、またはプロジェクトの所有者は、プロジェクトの公開範囲を設定および編集することができます。

詳細については、[プロジェクトの公開範囲]({{< relref path="./restricted-projects.md" lang="ja" >}})を参照してください。