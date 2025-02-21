---
title: Access management
cascade:
- url: guides/hosting/iam/access-management/:filename
menu:
  default:
    identifier: ja-guides-hosting-iam-access-management-_index
    parent: identity-and-access-management-iam
url: guides/hosting/iam/access-management-intro
weight: 2
---

## Manage users and teams within an organization
ユニークな組織ドメインでW&Bに最初にサインアップしたユーザーは、その組織の *インスタンス管理者ロール* として割り当てられます。組織管理者は特定のユーザーにチーム管理者ロールを割り当てます。

{{% alert %}}
W&Bは、組織に複数のインスタンス管理者を持つことを推奨しています。これは、主要な管理者が利用できない場合でも、管理者の操作を続けられるようにするためのベストプラクティスです。
{{% /alert %}}

*チーム管理者* は、組織内でチーム内の管理権限を持つユーザーです。

組織管理者は `https://wandb.ai/account-settings/` で組織のアカウント設定にアクセスし、ユーザーを招待したり、ユーザーのロールを割り当てまたは更新したり、チームを作成したり、組織からユーザーを削除したり、請求管理者を割り当てたりすることができます。詳細については、[Add and manage users]({{< relref path="./manage-organization.md#add-and-manage-users" lang="ja" >}}) を参照してください。

組織管理者がチームを作成すると、インスタンス管理者またはチーム管理者は以下を行うことができます：

- そのチームにユーザーを招待したり、チームからユーザーを削除したりします。
- チームメンバーのロールを割り当てたり更新したりします。
- 新しいユーザーが組織に参加したときに自動的にチームに追加します。

組織管理者とチーム管理者の両方が、`https://wandb.ai/<your-team-name>` のチームダッシュボードを使用してチームを管理します。組織管理者とチーム管理者ができることの詳細については、[Add and manage teams]({{< relref path="./manage-organization.md#add-and-manage-teams" lang="ja" >}}) を参照してください。

## Limit visibility to specific projects

W&B プロジェクトの範囲を定義して、誰がそのプロジェクトを表示、編集、および W&B Runs を送信できるかを制限します。プロジェクトを表示できる人を制限することは、チームが機密または秘密のデータを扱う場合に特に有用です。

組織管理者、チーム管理者、またはプロジェクトの所有者は、プロジェクトの可視性を設定および編集できます。

詳細については、[Project visibility]({{< relref path="./restricted-projects.md" lang="ja" >}}) を参照してください。