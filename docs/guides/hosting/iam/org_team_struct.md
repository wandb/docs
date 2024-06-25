---
displayed_sidebar: default
---


# IAM structure
W&B プラットフォームには、W&B 内に 3 つの IAM スコープがあります。[Organizations](#organization), [Teams](#team), [Projects](#project)です。

## Organization

*Organization* は、W&B アカウントまたはインスタンスのルートスコープです。アカウントまたはインスタンス内のすべての操作は、ユーザー管理、チーム管理、チーム内のプロジェクト管理、使用状況の追跡などを含む、このルートスコープのコンテキスト内で行われます。

[Multi-tenant Cloud](../hosting-options/saas_cloud.md) を使用している場合、それぞれが事業部、個人ユーザー、他のビジネスとの共同パートナーシップなどに対応する複数の organization を持つことがあります。

[Dedicated Cloud](../hosting-options/dedicated_cloud.md) もしくは [Self-managed instance](../hosting-options/self-managed.md) を使用している場合、これは 1 つの organization に対応します。貴社は異なる事業部や部署に対応する複数の Dedicated Cloud または Self-managed インスタンスを持つことができ、それは貴社内の AI プラクティショナーを管理するためのオプションの方法に過ぎません。

詳細は [Organizations](../../app/features/organizations.md) を参照してください。

## Team

*Team* は organization 内のサブスコープであり、企業内の事業部門/機能、部署、またはプロジェクトチームに対応することがあります。デプロイメントの種類や料金プランによっては、organization 内に複数のチームを持つことができます。

AI プロジェクトはチームのコンテキスト内で整理されます。チーム内のアクセスコントロールは、チームの管理者によって管理され、彼らは必ずしも親 organization レベルでの管理者とは限りません。

詳細は [Teams](../../app/features/teams.md) を参照してください。

## Project

*Project* はチーム内のサブスコープであり、特定の目的を持つ実際の AI プロジェクトに対応します。1 つのチーム内に複数のプロジェクトを持つことができます。各プロジェクトには、誰がアクセスできるかを決定する可視性モードがあります。

各プロジェクトは [Workspaces](../../app/pages/workspaces.md) と [Reports](../../reports/intro.md) で構成され、関連する [Artifacts](../../artifacts/intro.md)、[Sweeps](../../sweeps/intro.md)、[Launch Jobs](../../launch/intro.md)、および [Automations](../../artifacts/project-scoped-automations.md) とリンクされています。