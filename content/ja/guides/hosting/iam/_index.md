---
title: アイデンティティおよびアクセス管理 (IAM)
cascade:
- url: guides/hosting/iam/:filename
menu:
  default:
    identifier: ja-guides-hosting-iam-_index
    parent: w-b-platform
url: guides/hosting/iam/org_team_struct
weight: 2
---

W&B プラットフォームには、W&B 内に [Organizations]({{< relref path="#organization" lang="ja" >}})、[Teams]({{< relref path="#team" lang="ja" >}})、および [Projects]({{< relref path="#project" lang="ja" >}}) という 3 つの IAM スコープがあります。

## Organization

*Organization* は、W&B アカウントまたはインスタンスにおけるルート スコープです。アカウントまたはインスタンス内のすべてのアクションは、ユーザー管理、Teams 管理、Team 内の Projects 管理、使用状況の追跡など、このルート スコープのコンテキスト内で実行されます。

[マルチテナント クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) を使用している場合、複数の Organization を持つことができます。各 Organization は、事業単位、個人のユーザー、他社とのパートナーシップなどに対応する場合があります。

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) を使用している場合は、1 つの Organization に対応します。貴社では、異なる事業単位や部門にマッピングする目的で、複数の専用クラウド または Self-managed インスタンスを運用することもできますが、これは会社や部門全体の AI 実務者を管理するための必須要件ではありません。

詳細については、[Organization を管理する]({{< relref path="./access-management/manage-organization.md" lang="ja" >}}) を参照してください。

## Team

*Team* は、Organization 内のサブスコープであり、会社の事業単位 / 機能、部門、またはプロジェクト チームにマッピングできます。デプロイメント タイプと料金プランによって、Organization 内に複数の Team を持つことができます。

AI Projects は Team のコンテキスト内で整理されます。Team 内のアクセス制御は Team 管理者によって管理され、これらの管理者は親 Organization レベルの管理者である場合とそうでない場合があります。

詳細については、[Team を追加および管理する]({{< relref path="./access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) を参照してください。

## Project

*Project* は、Team 内のサブスコープであり、特定の意図された成果を持つ実際の AI プロジェクトにマッピングされます。1 つの Team 内に複数の Project を持つことができます。各 Project には公開範囲モードがあり、これによって誰がアクセスできるかが決まります。

すべての Project は [Workspaces]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) と [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) で構成され、関連する [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})、[Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})、および [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) にリンクされています。