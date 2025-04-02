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

W&B Platform には、W&B 内に3つの IAM スコープがあります。[Organizations]({{< relref path="#organization" lang="ja" >}})、[Teams]({{< relref path="#team" lang="ja" >}})、および [Projects]({{< relref path="#project" lang="ja" >}})です。

## Organization

*Organization* は、W&B アカウントまたはインスタンスのルートスコープです。アカウントまたはインスタンス内のすべてのアクションは、そのルートスコープのコンテキスト内で実行されます。これには、ユーザーの管理、Teams の管理、Teams 内の Projects の管理、使用状況の追跡などが含まれます。

[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) を使用している場合、複数の Organization を持つことができ、それぞれが事業部門、個人のユーザー、他の企業との共同パートナーシップなどに対応する場合があります。

[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed instance]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) を使用している場合は、1つの Organization に対応します。貴社は、事業部門や部署に対応するために、複数の Dedicated Cloud または Self-managed instance を持つことができますが、これは厳密には、貴社のビジネスまたは部署全体で AI 実務者を管理するためのオプションの方法です。

詳細については、[Organization の管理]({{< relref path="./access-management/manage-organization.md" lang="ja" >}})を参照してください。

## Team

*Team* は、Organization 内のサブスコープであり、事業部門/機能、部署、または社内の project team に対応する場合があります。デプロイメントの種類と料金プランに応じて、Organization 内に複数の Team を持つことができます。

AI projects は、Team のコンテキスト内で編成されます。Team 内のアクセス制御は、親 Organization レベルの管理者である場合とそうでない場合がある Team 管理者によって管理されます。

詳細については、[Team の追加と管理]({{< relref path="./access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}})を参照してください。

## Project

*Project* は、Team 内のサブスコープであり、特定の意図された結果を持つ実際の AI project に対応します。Team 内に複数の project を持つことができます。各 project には、誰がアクセスできるかを決定する可視性モードがあります。

すべての project は、[Workspaces]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) と [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) で構成され、関連する [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})、[Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})、および [Automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) にリンクされています。
