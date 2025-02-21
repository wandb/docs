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

## 組織内で ユーザー と Teams を管理する
固有の組織ドメインで W&B にサインアップした最初の ユーザー は、その組織の「インスタンス管理者ロール」として割り当てられます。組織管理者は、特定の ユーザー に Team 管理者ロールを割り当てます。

{{% alert %}}
W&B では、組織内に複数のインスタンス管理者を持つことを推奨しています。プライマリアドミンの利用ができない場合でも、管理業務を継続できるようにするためのベストプラクティスです。
{{% /alert %}}

「Team 管理者」は、Team 内で管理権限を持つ組織内の ユーザー です。

組織管理者は、`https://wandb.ai/account-settings/` で組織のアカウント 設定 にアクセスして使用し、 ユーザー の招待、 ユーザー のロールの割り当てまたは更新、Teams の作成、組織からの ユーザー の削除、課金管理者の割り当てなどを行うことができます。詳細については、[ ユーザー の追加と管理]({{< relref path="./manage-organization.md#add-and-manage-users" lang="ja" >}})を参照してください。

組織管理者が Team を作成すると、インスタンス管理者または Team 管理者は次のことができます。

- その Team への ユーザー の招待、または Team からの ユーザー の削除。
- Team メンバーのロールの割り当てまたは更新。
- 新しい ユーザー が組織に参加したときに、自動的に Team に追加する。

組織管理者と Team 管理者は、`https://wandb.ai/<your-team-name>` の Team ダッシュボード を使用して、Teams を管理します。組織管理者と Team 管理者ができることの詳細については、[Teams の追加と管理]({{< relref path="./manage-organization.md#add-and-manage-teams" lang="ja" >}})を参照してください。

## 特定の Projects への可視性を制限する

W&B Project の範囲を定義して、誰が W&B の run を表示、編集、および送信できるかを制限します。Team が機密 データ を扱う場合、 Project を表示できる ユーザー を制限すると特に役立ちます。

組織管理者、Team 管理者、または Project の所有者は、 Project の可視性を設定および編集できます。

詳細については、[Project の可視性]({{< relref path="./restricted-projects.md" lang="ja" >}})を参照してください。
