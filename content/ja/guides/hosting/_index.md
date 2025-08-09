---
title: W&B プラットフォーム
menu:
  default:
    identifier: ja-guides-hosting-_index
no_list: true
weight: 6
---

W&B Platform は、[Core]({{< relref path="/guides/core" lang="ja" >}})、[Models]({{< relref path="/guides/models/" lang="ja" >}})、[Weave]({{< relref path="/guides/weave/" lang="ja" >}}) などの W&B 製品を支える基盤インフラストラクチャー、ツール、ガバナンスの土台です。

W&B Platform には、以下の 3 種類のデプロイメントオプションがあります。

* [W&B Multi-tenant Cloud]({{< relref path="#wb-multi-tenant-cloud" lang="ja" >}})
* [W&B Dedicated Cloud]({{< relref path="#wb-dedicated-cloud" lang="ja" >}})
* [W&B Customer-managed]({{< relref path="#wb-customer-managed" lang="ja" >}})

以下の責任分担表は、それぞれの主な違いをまとめたものです。

|                                      | Multi-tenant Cloud                              | Dedicated Cloud                                                                      | Customer-managed             |
|--------------------------------------|------------------------------------------------|--------------------------------------------------------------------------------------|------------------------------|
| MySQL / DB 管理                     | W&B により完全ホスト・管理                      | 顧客が選択したクラウドやリージョンで W&B が完全ホスト・管理                          | 顧客による完全なホストと管理 |
| オブジェクトストレージ (S3/GCS/Blob) | **オプション 1**: W&B により完全ホスト<br />**オプション 2**: [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を利用して、チームごとに顧客がバケットを設定可能 | **オプション 1**: W&B により完全ホスト<br />**オプション 2**: [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を利用して、インスタンスまたはチームごとに顧客がバケットを設定可能 | 顧客による完全なホストと管理     |
| SSO サポート                        | Auth0 を利用した W&B 管理                       | **オプション 1**: 顧客管理<br />**オプション 2**: Auth0 経由で W&B が管理           | 顧客による完全管理            |
| W&B サービス (アプリ)                | W&B により完全管理                              | W&B により完全管理                                                                   | 顧客による完全管理            |
| アプリのセキュリティ                | W&B により完全管理                              | W&B と顧客の共同責任                                                                 | 顧客による完全管理            |
| 保守（アップグレード、バックアップ等）| W&B による管理                                  | W&B による管理                                                                       | 顧客による管理                |
| サポート                            | サポート SLA                                    | サポート SLA                                                                         | サポート SLA                  |
| サポートクラウドインフラ             | GCP                                            | AWS, GCP, Azure                                                                      | AWS, GCP, Azure, オンプレベアメタル |

## デプロイメントオプション
以下のセクションでは、各デプロイメントタイプの概要を紹介します。

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud は、W&B のクラウドインフラストラクチャー上に構築された完全マネージドサービスです。ご希望のスケールで W&B 製品にシームレスにアクセスでき、コスト効率の良い価格設定や、常に最新機能・機能強化への継続的なアップデートが提供されます。プライベートなデプロイメントのセキュリティが不要な場合や、セルフサービスでの利用開始やコスト効率が重要な場合のトライアルや本番 AI ワークフローには、Multi-tenant Cloud の利用をおすすめします。

さらに詳しくは [W&B Multi-tenant Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) をご覧ください。

### W&B Dedicated Cloud
W&B Dedicated Cloud は、W&B のクラウドインフラストラクチャー上に配置されるシングルテナント型の完全マネージドサービスです。データレジデンシーなどの厳格なガバナンス要件や高度なセキュリティ機能が必要な組織、インフラストラクチャーの構築・管理にかかるコストを抑えつつAI運用コストを最適化したい場合に最適です。

さらに詳しくは [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) をご覧ください。

### W&B Customer-Managed
このオプションでは、W&B Server を自身の管理するインフラストラクチャー上にデプロイし管理できます。W&B Server は、W&B Platform とそのサポート対象の W&B 製品を稼働させる自己完結型のパッケージとして提供されます。既存のインフラのすべてがオンプレであったり、W&B Dedicated Cloud では満たせない厳格な規制要件がある場合は、このオプションがおすすめです。この場合、インフラの調達や、継続的なメンテナンス・アップグレードに関してもすべてご自身で管理いただく必要があります。

さらに詳しくは [W&B Self Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) をご覧ください。

## 次のステップ

W&B 製品をお試しになりたい方には、[Multi-tenant Cloud](https://wandb.ai/home) のご利用がおすすめです。エンタープライズ向けの導入を希望される場合は、トライアルに最適なデプロイメントタイプを [こちら](https://wandb.ai/site/enterprise-trial) からお選びください。