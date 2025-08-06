---
title: W&B プラットフォーム
menu:
  default:
    identifier: w-b-platform
weight: 6
no_list: true
---

W&B Platform は、[Core]({{< relref "/guides/core" >}})、[Models]({{< relref "/guides/models/" >}})、[Weave]({{< relref "/guides/weave/" >}}) などの W&B 製品を支える基盤インフラストラクチャー、ツール群、ガバナンスの枠組みです。

W&B Platform は、以下の 3 つのデプロイメントオプションをご用意しています。

* [W&B Multi-tenant Cloud]({{< relref "#wb-multi-tenant-cloud" >}})
* [W&B Dedicated Cloud]({{< relref "#wb-dedicated-cloud" >}})
* [W&B Customer-managed]({{< relref "#wb-customer-managed" >}})

以下の責任分担マトリクスで各オプションの主な違いをまとめています。

|                                      | Multi-tenant Cloud                | Dedicated Cloud                                                     | Customer-managed |
|--------------------------------------|-----------------------------------|---------------------------------------------------------------------|------------------|
| MySQL / DB 管理                      | W&B により完全ホスト・管理         | W&B によりクラウドや選択リージョンで完全ホスト・管理               | お客様による完全ホスト・管理         |
| オブジェクトストレージ（S3/GCS/Blob など）| **オプション1**: W&B が完全ホスト<br />**オプション2**: [Secure Storage Connector]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) を利用し、チームごとにお客様が独自バケット設定可能 | **オプション1**: W&B が完全ホスト<br />**オプション2**: [Secure Storage Connector]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) を利用し、インスタンスまたはチームごとにお客様が独自バケット設定可能 | お客様による完全ホスト・管理         |
| SSO サポート                         | W&B が Auth0 経由で管理            | **オプション1**: お客様による管理<br />**オプション2**: W&B が Auth0 経由で管理 | お客様による完全管理               |
| W&B サービス（アプリ）                | W&B による完全管理                  | W&B による完全管理                                                  | お客様による完全管理               |
| アプリのセキュリティ                  | W&B による完全管理                  | W&B とお客様の共同責任                                              | お客様による完全管理               |
| 保守（アップグレード・バックアップ等）| W&B が管理                          | W&B が管理                                                           | お客様が管理                       |
| サポート                             | サポート SLA                       | サポート SLA                                                         | サポート SLA                       |
| サポートされるクラウドインフラストラクチャー | GCP                               | AWS, GCP, Azure                                                     | AWS, GCP, Azure, オンプレミス ベアメタル |

## デプロイメントオプション
以下のセクションで各デプロイメントタイプの概要をご紹介します。

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud は、W&B のクラウドインフラストラクチャー上で提供される完全マネージド型サービスです。用途に応じたスケールで W&B 製品にシームレスにアクセスでき、コスト効率の高い料金体系や、最新機能・アップデートが継続的に利用できます。Multi-tenant Cloud は、お試し利用や、プライベートなデプロイメントのセキュリティ要件が特に不要で、セルフサービスでの導入や高いコスト効率が重要な場合のプロダクション AI ワークフローの運用におすすめです。

詳細は [W&B Multi-tenant Cloud]({{< relref "./hosting-options/saas_cloud.md" >}}) をご覧ください。

### W&B Dedicated Cloud
W&B Dedicated Cloud は、W&B のクラウドインフラストラクチャー上に展開されるシングルテナント型の完全マネージドサービスです。データレジデンシー（データ所在地等）を含む厳格なガバナンス要件への準拠が求められる場合、高度なセキュリティ機能が必要な場合、あるいはセキュリティ・スケール・パフォーマンスを社内で作りこむことなく AI の運用コストを最適化したい場合に最適な選択肢です。

詳細は [W&B Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) をご覧ください。

### W&B Customer-Managed
このオプションでは、ご自身で管理しているインフラストラクチャー上に W&B Server をデプロイし運用できます。W&B Server は、W&B Platform および対応する W&B 製品を動かすための自己完結型パッケージです。既存のインフラストラクチャーがすべてオンプレミスの場合や、W&B Dedicated Cloud でカバーできない厳しい規制要件が組織にある場合に推奨します。この場合、W&B Server を支えるインフラストラクチャーの調達・継続的な保守やアップグレード等をお客様がすべて管理いただく必要があります。

詳細は [W&B Self Managed]({{< relref "/guides/hosting/hosting-options/self-managed/" >}}) をご覧ください。

## 次のステップ

W&B 製品をお試しの場合は、[Multi-tenant Cloud](https://wandb.ai/home) のご利用をおすすめします。エンタープライズ向けの環境が必要な場合は、[こちら](https://wandb.ai/site/enterprise-trial) からご希望のデプロイメントタイプでトライアルをお申し込みください。