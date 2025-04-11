---
title: W&B プラットフォーム
menu:
  default:
    identifier: ja-guides-hosting-_index
no_list: true
weight: 6
---

W&B Platformは、W&Bの製品である[Core]({{< relref path="/guides/core" lang="ja" >}})、[Models]({{< relref path="/guides/models/" lang="ja" >}})、および[Weave]({{< relref path="/guides/weave/" lang="ja" >}})をサポートするための基本となるインフラストラクチャー、ツール、およびガバナンスの枠組みです。

W&B Platformは、以下の3つの異なるデプロイメントオプションで利用可能です:

* [W&B Multi-tenant Cloud]({{< relref path="#wb-multi-tenant-cloud" lang="ja" >}})
* [W&B Dedicated Cloud]({{< relref path="#wb-dedicated-cloud" lang="ja" >}})
* [W&B Customer-managed]({{< relref path="#wb-customer-managed" lang="ja" >}})

以下の責任分担表は、いくつかの主な違いを示しています:

|                                      | Multi-tenant Cloud                        | Dedicated Cloud                                                                 | Customer-managed |
|--------------------------------------|-------------------------------------------|---------------------------------------------------------------------------------|------------------|
| MySQL / DB management                | 完全にW&Bがホストおよび管理               | 完全にW&Bが顧客が選択したクラウドまたはリージョンでホストおよび管理             | 完全に顧客がホストおよび管理 |
| Object Storage (S3/GCS/Blob storage) | **オプション 1**: W&Bが完全にホスト<br />**オプション 2**: [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を使用して、顧客がチームごとに独自のバケットを設定可能 | **オプション 1**: W&Bが完全にホスト<br />**オプション 2**: [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を使用して、顧客がインスタンスまたはチームごとに独自のバケットを設定可能 | 完全に顧客がホストおよび管理 |
| SSO Support                          | Auth0を通じてW&Bが管理                   | **オプション 1**: 顧客が管理<br />**オプション 2**: Auth0を通じてW&Bが管理       | 完全に顧客が管理   |
| W&B Service (App)                    | W&Bが完全に管理                          | W&Bが完全に管理                                                                 | 完全に顧客が管理         |
| App security                         | W&Bが完全に管理                          | W&Bと顧客の共同責任                                                             | 完全に顧客が管理         |
| Maintenance (upgrades, backups, etc.)| W&Bが管理                               | W&Bが管理                                                                      | 顧客が管理 |
| Support                              | サポートSLA                             | サポートSLA                                                                     | サポートSLA |
| Supported cloud infrastructure       | GCP                                     | AWS, GCP, Azure                                                                | AWS, GCP, Azure, オンプレミスベアメタル |

## デプロイメントオプション
以下のセクションでは、各デプロイメントタイプの概要を示します。

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloudは、W&Bのクラウドインフラストラクチャーにデプロイされた完全管理型サービスで、希望の規模でW&Bの製品にシームレスにアクセスでき、価格設定のための費用対効果の高いオプションや、最新の機能と機能性のための継続的なアップデートを提供します。W&Bは、Multi-tenant Cloudを製品トライアルに使用することを推奨しています。また、プライベートデプロイメントのセキュリティが不要で、自己サービスのオンボーディングが重要であり、コスト効率が重要である場合には、プロダクションAIワークフローを管理するのに適しています。

詳細は、[W&B Multi-tenant Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}})をご覧ください。

### W&B Dedicated Cloud
W&B Dedicated Cloudは、W&Bのクラウドインフラストラクチャーにデプロイされるシングルテナントの完全管理型サービスです。データレジデンシーを含む厳格なガバナンス管理に準拠する必要があり、高度なセキュリティ機能を求め、セキュリティ、スケール、性能特性を備えた必要なインフラストラクチャーを構築・管理することなくAI運用コストを最適化しようとしている場合、W&Bを導入する最適な場所です。

詳細は、[W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}})をご覧ください。

### W&B Customer-Managed
このオプションでは、独自に管理するインフラストラクチャーにW&B Serverをデプロイし、管理することができます。W&B Serverは、W&B Platformとそのサポート製品を実行するための自己完結型パッケージメカニズムです。W&Bは、既存のすべてのインフラストラクチャーがオンプレミスである場合、またはW&B Dedicated Cloudでは満たされない厳格な規制ニーズがある場合、このオプションを推奨します。このオプションでは、W&B Serverをサポートするために必要なインフラストラクチャーのプロビジョニング、および継続的なメンテナンスとアップグレードを管理する完全な責任を負います。

詳細は、[W&B Self Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}})をご覧ください。

## 次のステップ

W&Bの製品を試したい場合、W&Bは[Multi-tenant Cloud](https://wandb.ai/home)の使用を推奨します。企業向けのセットアップを探している場合は、[こちら](https://wandb.ai/site/enterprise-trial)からトライアルに適したデプロイメントタイプを選択してください。