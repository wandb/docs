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

## 組織内の ユーザー と Teams を管理する
一意の組織ドメインで W&B に最初にサインアップした ユーザー は、その組織の _インスタンス管理者ロール_ として割り当てられます。組織管理者は、特定の ユーザー に チーム 管理者ロールを割り当てます。

{{% alert %}}
W&B では、組織内に複数のインスタンス管理者を持つことを推奨します。これは、プライマリアドミンが利用できない場合でも、管理業務を継続できるようにするためのベストプラクティスです。
{{% /alert %}}

_チーム管理者_ は、 チーム 内で管理権限を持つ組織内の ユーザー です。

組織管理者は、`https://wandb.ai/account-settings/` で組織のアカウント 設定 にアクセスして使用し、 ユーザー の招待、 ユーザー のロールの割り当てまたは更新、 Teams の作成、組織からの ユーザー の削除、請求管理者の割り当てなどを行うことができます。詳細については、[ユーザー の追加と管理]({{< relref path="./manage-organization.md#add-and-manage-users" lang="ja" >}})を参照してください。

組織管理者が チーム を作成すると、インスタンス管理者または チーム 管理者は次のことができます。

- デフォルトでは、管理者のみがその チーム に ユーザー を招待したり、 チーム から ユーザー を削除したりできます。この 振る舞い を変更するには、[チーム の 設定 ]({{< relref path="/guides/models/app/settings-page/team-settings.md#privacy" lang="ja" >}})を参照してください。
- チームメンバー のロールを割り当てるか、更新します。
- 新しい ユーザー が組織に参加したときに、自動的に ユーザー を チーム に追加します。

組織管理者と チーム 管理者の両方が、`https://wandb.ai/<your-team-name>` の チーム ダッシュボード を使用して Teams を管理します。詳細、および チーム のデフォルトのプライバシー 設定 の構成については、[Teams の追加と管理]({{< relref path="./manage-organization.md#add-and-manage-teams" lang="ja" >}})を参照してください。

## 特定の Projects への可視性を制限する

W&B の Project のスコープを定義して、誰が W&B の Runs を表示、編集、および送信できるかを制限します。 チーム が機密 データ を扱う場合、 Project を表示できる ユーザー を制限すると特に役立ちます。

組織管理者、 チーム 管理者、または Project のオーナーは、 Project の可視性を 設定 および編集できます。

詳細については、[Project の可視性]({{< relref path="./restricted-projects.md" lang="ja" >}})を参照してください。
