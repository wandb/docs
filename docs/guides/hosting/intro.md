---
slug: /guides/hosting
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# W&B Platform

W&B Platformは、[Core](../platform.md)、[Models](../models.md)、[Prompts](../prompts_platform.md)といったW&B製品を支える基盤インフラストラクチャー、ツール、およびガバナンスの枠組みです。

W&B Platformは、以下の3つのデプロイメントオプションで利用可能です:
* [W&B Multi-tenant Cloud](#wb-saas-cloud)
* [W&B Dedicated Cloud](#wb-dedicated-cloud)
* [W&B Customer-managed](#wb-customer-managed)

以下の責任分担マトリックスは、各オプションの主な違いを示しています:
![](/images/hosting/shared_responsibility_matrix.png)

## デプロイメントオプション
以下のセクションでは、各デプロイメントタイプの概要を説明します。

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloudは、W&Bのクラウドインフラストラクチャーにデプロイされた完全に管理されたサービスで、必要なスケールでW&B製品にシームレスにアクセスでき、コスト効率の良い価格オプションを提供し、最新の機能や機能の継続的な更新を受け取ることができます。W&Bは、プライベートなデプロイメントのセキュリティが必要なく、セルフサービスのオンボーディングが重要で、コスト効率が重要な場合には、Multi-tenant Cloudを製品トライアルやプロダクションAIワークフローの管理に使用することを推奨しています。

詳細については、[W&B Multi-tenant Cloud](./hosting-options/saas_cloud.md)を参照してください。

### W&B Dedicated Cloud
W&B Dedicated Cloudは、W&Bのクラウドインフラストラクチャーにデプロイされたシングルテナントの完全に管理されたサービスです。データレジデンシーを含む厳格なガバナンスコントロールに従う必要がある場合、また、高度なセキュリティ機能が必要で、セキュリティ、スケール、パフォーマンス特性を持つインフラストラクチャーを構築・管理することなくAI運用コストを最適化しようとする場合に、W&Bを導入する最適な選択肢です。

詳細については、[W&B Dedicated Cloud](./hosting-options/dedicated_cloud.md)を参照してください。

### W&B Customer-Managed
このオプションを使用すると、独自の管理インフラストラクチャーでW&B Serverをデプロイおよび管理できます。W&B Serverは、W&B PlatformとそれがサポートするW&B製品を実行するための自己完結型のパッケージメカニズムです。既存のすべてのインフラストラクチャーがオンプレミスである場合、またはW&B Dedicated Cloudで満たされない厳格な規制要件がある場合には、このオプションを推奨します。このオプションを使用すると、W&B Serverをサポートするために必要なインフラストラクチャーのプロビジョニング、および継続的なメンテナンスとアップグレードの管理が完全にあなたの責任になります。

詳細については、[W&B Self Managed](./hosting-options/self-managed.md)を参照してください。

## 次のステップ

任意のW&B製品を試したい場合は、W&Bは[Multi-tenant Cloud](https://wandb.ai/home)を使用することを推奨します。エンタープライズ向けのセットアップを希望する場合は、[ここ](https://wandb.ai/site/enterprise-trial)でトライアルに適したデプロイメントタイプを選択してください。