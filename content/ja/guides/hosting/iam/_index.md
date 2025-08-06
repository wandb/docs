---
title: Identity および アクセス管理 (IAM)
cascade:
- url: guides/hosting/iam/:filename
menu:
  default:
    identifier: ja-guides-hosting-iam-_index
    parent: w-b-platform
url: guides/hosting/iam/org_team_struct
weight: 2
---

W&B プラットフォームには、W&B 内での IAM スコープが3つあります: [Organizations]({{< relref path="#organization" lang="ja" >}})、[Teams]({{< relref path="#team" lang="ja" >}})、そして [Projects]({{< relref path="#project" lang="ja" >}}) です。

## Organization

*Organization* は、W&B アカウントまたはインスタンスにおける最上位のスコープです。アカウントまたはインスタンス内のすべての操作（ユーザー管理、チーム管理、チーム内プロジェクト管理、利用状況のトラッキングなど）は、この最上位スコープのコンテキスト内で行われます。

[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) をご利用の場合、複数の organization を持つことができます。それぞれが事業部門、個人ユーザー、他のビジネスとの共同プロジェクトなどに対応している場合があります。

[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed instance]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) を利用する場合、1つの organization に対応します。企業ごとに複数の Dedicated Cloud や Self-managed instance を導入し、異なる事業部門や部署ごとに分けて運用することもできますが、これはあくまでオプションの運用方法です。

詳細は、[組織の管理]({{< relref path="./access-management/manage-organization.md" lang="ja" >}}) をご覧ください。

## Team

*Team* は organization 内のサブスコープであり、事業部門／機能、部署、プロジェクトチームに対応します。導入形態や料金プランに応じて、1つの organization 内に複数の team を構成できます。

AI プロジェクトは team のコンテキスト内で整理されます。team 内でのアクセスコントロールは、その team の管理者によって行われ、organization 全体の管理者とは限りません。

詳しくは [チームの追加・管理方法]({{< relref path="./access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) をご覧ください。

## Project

*Project* は team 内のサブスコープで、実際の AI プロジェクトやその成果物に対応します。1つの team 内に複数の project を持つことも可能です。各 project には、誰がアクセスできるかを決める公開範囲（visibility）モードがあります。

各 project は [Workspaces]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) と [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) で構成され、さらに関連する [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})、[Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})、[Automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) と連携しています。