---
title: アイデンティティとアクセス管理 (IAM)
cascade:
- url: guides/hosting/iam/:filename
menu:
  default:
    identifier: ja-guides-hosting-iam-_index
    parent: w-b-platform
url: guides/hosting/iam/org_team_struct
weight: 2
---

W&B プラットフォームには、W&B 内での 3 つの IAM スコープがあります: [Organizations]({{< relref path="#organization" lang="ja" >}})、[Teams]({{< relref path="#team" lang="ja" >}})、および [Projects]({{< relref path="#project" lang="ja" >}})。

## Organization

*Organization* は、あなたの W&B アカウントまたはインスタンスのルートスコープです。ユーザー管理、チーム管理、チーム内のプロジェクト管理、使用状況の追跡など、アカウントまたはインスタンス内のすべてのアクションは、このルートスコープのコンテキスト内で行われます。

[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) を使用している場合、複数の組織を持っている可能性があります。それぞれが事業部門、個人のユーザー、他社との共同パートナーシップなどに対応する場合があります。

[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed instance]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) を使用している場合、それは1つの組織に対応しています。あなたの会社は、異なる事業部門または部門に対応するために Dedicated Cloud または Self-managed インスタンスを複数持つことができますが、それは業務や部門全体にわたる AI 実践者を管理するためのオプションの方法に過ぎません。

詳細については、[Manage organizations]({{< relref path="./access-management/manage-organization.md" lang="ja" >}}) を参照してください。

## Team

*Team* は、組織内のサブスコープであり、事業部門、機能、部門、または会社内のプロジェクトチームに対応する場合があります。デプロイメントタイプと価格プランに応じて、組織内に複数のチームを持つことができます。

AI プロジェクトはチームのコンテキスト内で編成されます。チーム内のアクセス制御は、親組織レベルで管理者であるかどうかに関係なく、チーム管理者によって管理されます。

詳細については、[Add and manage teams]({{< relref path="./access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) を参照してください。

## Project

*Project* は、特定の目標を持つ実際の AI プロジェクトに対応するチーム内のサブスコープです。チーム内に複数のプロジェクトを持つことができます。各プロジェクトには、誰がプロジェクトにアクセスできるかを決定する公開範囲モードがあります。

各プロジェクトは、[Workspaces]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) と [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) で構成され、関連する [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})、[Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})、および [Automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) とリンクされています。