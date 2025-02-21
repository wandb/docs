---
title: W&B Platform
menu:
  default:
    identifier: ja-guides-hosting-_index
no_list: true
weight: 6
---

W&B Platform は、[Core]({{< relref path="/guides/core/" lang="ja" >}})、[Models]({{< relref path="/guides/models/" lang="ja" >}}) 、[Weave]({{< relref path="/guides/weave/" lang="ja" >}}) などの W&B 製品をサポートする、基盤となるインフラストラクチャー、 ツール 、およびガバナンスの足場です。

W&B Platform は、次の3つの異なる デプロイメント オプションで利用できます。

* [W&B Multi-tenant Cloud]({{< relref path="#wb-multi-tenant-cloud" lang="ja" >}})
* [W&B Dedicated Cloud]({{< relref path="#wb-dedicated-cloud" lang="ja" >}})
* [W&B Customer-managed]({{< relref path="#wb-customer-managed" lang="ja" >}})

次の責任分担表は、異なるオプション間の主な違いのいくつかを示しています。
{{< img src="/images/hosting/shared_responsibility_matrix.png" alt="" >}}

## デプロイメント のオプション
以下のセクションでは、各 デプロイメント タイプ の概要を説明します。

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud は、W&B の クラウド インフラストラクチャー に デプロイ されたフルマネージド サービスであり、手頃な価格設定のオプションと最新の機能の継続的な更新により、希望する規模で W&B 製品にシームレスに アクセス できます。プライベート デプロイメント のセキュリティを必要とせず、セルフサービスでのオンボーディングが重要で、コスト効率が重要な場合は、 製品 の トライアル 、または プロダクション AI ワークフロー の管理に Multi-tenant Cloud を使用することをお勧めします。

詳細については、[W&B Multi-tenant Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) を参照してください。

### W&B Dedicated Cloud
W&B Dedicated Cloud は、W&B の クラウド インフラストラクチャー に デプロイ された シングルテナント のフルマネージド サービスです。 データ レジデンシー を含む厳格なガバナンス コントロール への準拠が組織で必要とされ、高度なセキュリティ機能を必要とし、セキュリティ、スケール、およびパフォーマンスの特性を備えた必要な インフラストラクチャー を構築および管理することなく、AI 運用コストを最適化したい場合は、W&B のオンボーディングに最適です。

詳細については、[W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) を参照してください。

### W&B Customer-Managed
このオプションを使用すると、独自のマネージド インフラストラクチャー 上で W&B Server を デプロイ および管理できます。W&B Server は、W&B Platform とそのサポートされている W&B 製品を実行するための自己完結型のパッケージ化されたメカニズムです。既存の インフラストラクチャー がすべて オンプレミス である場合、または組織が W&B Dedicated Cloud では満たされない厳格な規制ニーズを持っている場合は、このオプションをお勧めします。このオプションを使用すると、W&B Server をサポートするために必要な インフラストラクチャー のプロビジョニング、および継続的なメンテナンスとアップグレードを管理する責任を完全に負います。

詳細については、[W&B Self Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) を参照してください。

## 次のステップ

W&B 製品を試してみたい場合は、[Multi-tenant Cloud](https://wandb.ai/home) の使用をお勧めします。エンタープライズ向けのセットアップをお探しの場合は、[こちら](https://wandb.ai/site/enterprise-trial) で トライアル に適した デプロイメント タイプ を選択してください。
