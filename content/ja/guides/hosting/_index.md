---
title: W&B Platform
menu:
  default:
    identifier: ja-guides-hosting-_index
no_list: true
weight: 6
---

W&B Platform は、[Core]({{< relref path="/guides/core/" lang="ja" >}})、[Models]({{< relref path="/guides/models/" lang="ja" >}})、[Weave]({{< relref path="/guides/weave/" lang="ja" >}}) などの W&B 製品をサポートする基盤インフラストラクチャー、ツール、ガバナンスの足場です。

W&B Platform は、次の 3 つの異なるデプロイメントオプションで利用できます。

* [W&B Multi-tenant Cloud]({{< relref path="#wb-multi-tenant-cloud" lang="ja" >}})
* [W&B Dedicated Cloud]({{< relref path="#wb-dedicated-cloud" lang="ja" >}})
* [W&B Customer-managed]({{< relref path="#wb-customer-managed" lang="ja" >}})

以下の責任分担マトリックスは、異なるオプション間のいくつかの重要な違いを示しています。
{{< img src="/images/hosting/shared_responsibility_matrix.png" alt="" >}}

## デプロイメントオプション
以下のセクションでは、各デプロイメントタイプの概要を提供します。

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud は、W&B のクラウド インフラストラクチャーにデプロイされたフル マネージド サービスで、W&B 製品にシームレスにアクセスでき、希望の規模でコスト効率の高い価格オプションを提供し、最新の機能と特長のために継続的に更新されます。プライベートデプロイメントのセキュリティを必要とせず、セルフサービスオンボーディングが重要で、コスト効率がクリティカルである場合に、製品のトライアルやプロダクション AI ワークフローの管理には、Multi-tenant Cloud を使用することを W&B は推奨します。

詳細は [W&B Multi-tenant Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) を参照してください。

### W&B Dedicated Cloud
W&B Dedicated Cloud は、W&B のクラウド インフラストラクチャーにデプロイされたシングルテナントのフル マネージド サービスです。データの所在管理、厳密なガバナンスコントロールに従う必要があり、高度なセキュリティ機能を備えた AI 運用コストを最適化したい場合には、W&B に最適の導入場所です。

詳細は [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) を参照してください。

### W&B Customer-Managed
このオプションでは、独自の管理インフラストラクチャー上で W&B Server をデプロイおよび管理できます。W&B Server は、W&B Platform とそのサポート対象 W&B 製品を実行するためのパッケージ化されたメカニズムです。既存のインフラストラクチャーがすべてオンプレミスであるか、または W&B Dedicated Cloud では満たされない厳しい規制要件がある場合には、このオプションを W&B は推奨します。このオプションを選択した場合、W&B Server をサポートするために必要なインフラストラクチャーのプロビジョニング、継続的な保守、アップグレードを管理する完全な責任があります。

詳細は [W&B Self Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) を参照してください。

## 次のステップ

W&B のいずれかの製品をお試しになりたい場合、W&B は [Multi-tenant Cloud](https://wandb.ai/home) を使用することをお勧めします。エンタープライズ向けのセットアップが必要な場合は、[こちら](https://wandb.ai/site/enterprise-trial) で適切なデプロイメントタイプを選んでトライアルください。