---
title: Identity and access management (IAM)
cascade:
- url: guides/hosting/iam/:filename
menu:
  default:
    identifier: ja-guides-hosting-iam-_index
    parent: w-b-platform
url: guides/hosting/iam/org_team_struct
weight: 2
---

W&B プラットフォームには、3 つの IAM スコープがあります: [Organizations]({{< relref path="#organization" lang="ja" >}})、[Teams]({{< relref path="#team" lang="ja" >}})、[Projects]({{< relref path="#project" lang="ja" >}})。

## Organization

*Organization* は、W&B アカウントまたはインスタンスのルートスコープです。アカウントまたはインスタンスでのすべての操作は、このルートスコープ内で行われ、ユーザー管理、チーム管理、チーム内プロジェクト管理、使用状況の追跡などが含まれます。

[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) を使用している場合、複数の Organization を持つことができ、それぞれがビジネスユニット、個人ユーザー、他のビジネスとの共同パートナーシップなどに対応する場合があります。

[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) を使用している場合、それは 1 つの Organization に対応します。御社は、ビジネスユニットや部門ごとに異なる Dedicated Cloud または Self-managed インスタンスを保持することができ、これは異なるビジネスや部門にまたがる AI プラクティショナーを管理するためのオプションの方法です。

詳細については、[Manage organizations]({{< relref path="./access-management/manage-organization.md" lang="ja" >}}) を参照してください。

## Team

*Team* は、Organization 内のサブスコープであり、ビジネスユニット/機能、部門、またはプロジェクトチームに対応します。デプロイメントタイプやプランに応じて、Organization 内に複数のチームを持つことがあります。

AI プロジェクトは、チームのスコープ内に整理されます。チーム内のアクセス制御は、チームの管理者によって管理され、これらの管理者は上位の Organization レベルの管理者である場合とそうでない場合があります。

詳細については、[Add and manage teams]({{< relref path="./access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) を参照してください。

## Project

*Project* はチーム内のサブスコープであり、具体的な成果を意図した実際の AI プロジェクトに対応します。チーム内には複数のプロジェクトを持つことができます。各プロジェクトには、誰がそのプロジェクトに アクセス できるかを決定する可視性モードがあります。

各プロジェクトは [Workspaces]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) と [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) で構成され、関連する [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})、[Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})、[Launch Jobs]({{< relref path="/launch/" lang="ja" >}}) および [Automations]({{< relref path="/guides/models/automations/project-scoped-automations.md" lang="ja" >}}) にリンクされています。