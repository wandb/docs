---
title: W&B Platform
menu:
  default:
    identifier: ja-guides-hosting-_index
no_list: true
weight: 6
---

W&B Platform は、[Core]({{< relref path="/guides/core" lang="ja" >}})、[Models]({{< relref path="/guides/models/" lang="ja" >}})、[Weave]({{< relref path="/guides/weave/" lang="ja" >}}) などの W&B 製品をサポートする、基盤となるインフラストラクチャー、 ツール 、およびガバナンスの足場です。

W&B Platform は、次の3つの異なる デプロイメント オプションで利用できます。

* [W&B Multi-tenant Cloud]({{< relref path="#wb-multi-tenant-cloud" lang="ja" >}})
* [W&B Dedicated Cloud]({{< relref path="#wb-dedicated-cloud" lang="ja" >}})
* [W&B Customer-managed]({{< relref path="#wb-customer-managed" lang="ja" >}})

次の責任分担表は、主な違いの概要を示しています。

|                                      | Multi-tenant Cloud                | Dedicated Cloud                                                     | Customer-managed |
|--------------------------------------|-----------------------------------|---------------------------------------------------------------------|------------------|
| MySQL / DB 管理                | W&B が完全にホストおよび管理     | W&B が クラウド 上またはお客様が選択したリージョンで完全にホストおよび管理 | お客様が完全にホストおよび管理 |
| オブジェクトストレージ (S3/GCS/Blob storage) | **オプション1**: W&B が完全にホスト<br />**オプション2**: お客様は、[Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を使用して、 チーム ごとに独自の バケット を構成できます | **オプション1**: W&B が完全にホスト<br />**オプション2**: お客様は、[Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を使用して、インスタンスまたは チーム ごとに独自の バケット を構成できます | お客様が完全にホストおよび管理 |
| SSO サポート                          | Auth0 経由で W&B が管理             | **オプション1**: お客様が管理<br />**オプション2**: Auth0 経由で W&B が管理 | お客様が完全に管理   |
| W&B サービス (App)                    | W&B が完全に管理              | W&B が完全に管理                                                | お客様が完全に管理          |
| App セキュリティー                         | W&B が完全に管理              | W&B とお客様の共同責任                           | お客様が完全に管理         |
| メンテナンス (アップグレード、 バックアップ など) | W&B が管理 | W&B が管理 | お客様が管理 |
| サポート                              | サポート SLA                       | サポート SLA                                                         | サポート SLA |
| サポートされている クラウド インフラストラクチャー       | GCP                               | AWS、GCP、Azure                                                     | AWS、GCP、Azure、 オンプレミス ベアメタル |

## デプロイメント オプション
次のセクションでは、各 デプロイメント タイプの概要について説明します。

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud は、W&B の クラウド インフラストラクチャー に デプロイ されたフルマネージド サービスです。ここでは、希望する規模で W&B 製品にシームレスに アクセス でき、費用対効果の高い価格オプション、最新の機能と機能の継続的なアップデートを利用できます。プライベート デプロイメント のセキュリティーが不要で、セルフサービスでのオンボーディングが重要であり、コスト効率が重要な場合は、製品 トライアル に Multi-tenant Cloud を使用するか、 プロダクション AI ワークフロー を管理することをお勧めします。

詳細については、[W&B Multi-tenant Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) を参照してください。

### W&B Dedicated Cloud
W&B Dedicated Cloud は、W&B の クラウド インフラストラクチャー に デプロイ された シングルテナント のフルマネージド サービスです。 データ 常駐を含む厳格なガバナンス コントロールへの準拠が組織で必要であり、高度なセキュリティー機能を必要とし、セキュリティー、スケール、およびパフォーマンスの特性を備えた必要な インフラストラクチャー を構築および管理する必要がないことによって AI の運用コストを最適化しようとしている場合は、W&B をオンボーディングするのに最適な場所です。

詳細については、[W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) を参照してください。

### W&B Customer-Managed
このオプションを使用すると、独自の管理対象 インフラストラクチャー に W&B Server を デプロイ して管理できます。W&B Server は、W&B Platform と、サポートされている W&B 製品を 実行 するための自己完結型のパッケージ化されたメカニズムです。既存の インフラストラクチャー がすべて オンプレミス にあり、W&B Dedicated Cloud では満たされない厳格な規制ニーズが組織にある場合は、このオプションをお勧めします。このオプションを使用すると、W&B Server をサポートするために必要な インフラストラクチャー のプロビジョニング、および継続的な メンテナンス とアップグレードを管理する責任を完全に負います。

詳細については、[W&B Self Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) を参照してください。

## 次のステップ

W&B 製品のいずれかを試してみたい場合は、[Multi-tenant Cloud](https://wandb.ai/home) を使用することをお勧めします。エンタープライズ向けのセットアップをお探しの場合は、 トライアル に適した デプロイメント タイプを[こちら](https://wandb.ai/site/enterprise-trial)から選択してください。
