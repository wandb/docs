---
title: アイデンティティとアクセス管理（IAM）
menu:
  default:
    identifier: identity-and-access-management-iam
    parent: w-b-platform
url: guides/hosting/iam/org_team_struct
cascade:
- url: guides/hosting/iam/:filename
weight: 2
---

W&B プラットフォームでは、W&B 内に 3 つの IAM スコープがあります: [Organizations]({{< relref "#organization" >}})、[Teams]({{< relref "#team" >}})、および [Projects]({{< relref "#project" >}})。

## Organization

*Organization* は、あなたの W&B アカウントまたはインスタンスのルートスコープです。アカウントまたはインスタンス内でのすべての操作は、このルートスコープ内で行われ、ユーザー管理、チーム管理、チーム内でのプロジェクト管理、利用状況の追跡などを含みます。

[Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) を利用している場合、複数の organization を持つことができ、それぞれがビジネスユニット、個人ユーザー、他社との共同パートナーシップなどに対応します。

[Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) または [Self-managed instance]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) を利用している場合、1 つの organization に対応します。ただし、会社によっては複数の Dedicated Cloud または Self-managed instance を持ち、それぞれのビジネスユニットや部門に対応させることも可能です。これは AI 実践者を複数のビジネスや部門で管理するための、あくまでオプションの方法です。

詳細は [Manage organizations]({{< relref "./access-management/manage-organization.md" >}}) をご覧ください。

## Team

*Team* は organization 内のサブスコープであり、ビジネスユニット/機能、部門、または会社内のプロジェクトチームに対応します。使用しているデプロイメントタイプや料金プランによっては、organization 内に複数の Team を持つことができます。

AI プロジェクトは Team のコンテキスト内で整理されます。Team 内でのアクセス制御は team admin によって管理され、admin は親 organization の管理者である場合もない場合もあります。

詳細は [Add and manage teams]({{< relref "./access-management/manage-organization.md#add-and-manage-teams" >}}) をご覧ください。

## Project

*Project* は Team 内のサブスコープで、具体的な成果を目指す実際の AI プロジェクトに対応します。1 つの Team の中に複数の Project を持つことができます。各 Project には、誰がアクセスできるかを決定する visibility モードがあります。

すべての Project は [Workspaces]({{< relref "/guides/models/track/workspaces.md" >}}) と [Reports]({{< relref "/guides/core/reports/" >}}) で構成されており、関連する [Artifacts]({{< relref "/guides/core/artifacts/" >}})、[Sweeps]({{< relref "/guides/models/sweeps/" >}})、および [Automations]({{< relref "/guides/core/automations/" >}}) とリンクしています。