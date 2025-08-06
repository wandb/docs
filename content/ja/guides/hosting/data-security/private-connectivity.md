---
title: 専用クラウドへのプライベート接続を設定する
menu:
  default:
    identifier: ja-guides-hosting-data-security-private-connectivity
    parent: data-security
weight: 4
---

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) インスタンスには、クラウドプロバイダーのセキュアなプライベートネットワーク経由で接続できます。これは、AI ワークロードから W&B API へのアクセスや、オプションでユーザーブラウザから W&B アプリ UI へのアクセスにも適用されます。プライベート接続を利用すると、該当するリクエストやレスポンスはパブリックネットワークやインターネットを経由しません。

{{% alert %}}
セキュアなプライベート接続は、専用クラウド向けの高度なセキュリティオプションとして近日公開予定です。
{{% /alert %}}

セキュアなプライベート接続は、AWS・GCP・Azure 上の専用クラウドインスタンスでご利用いただけます:

* AWS では [AWS Privatelink](https://aws.amazon.com/privatelink/) の利用
* GCP では [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) の利用
* Azure では [Azure Private Link](https://azure.microsoft.com/products/private-link) の利用

有効化すると、W&B はお客様のインスタンス向けにプライベートエンドポイントサービスを作成し、接続に必要な DNS URI をご提供します。これにより、お客様のクラウドアカウント内でプライベートエンドポイントを作成し、該当トラフィックをプライベートエンドポイントサービスへルーティングできます。プライベートエンドポイントは、クラウド VPC または VNet 内で動作する AI トレーニングワークロード向けに、より簡単に設定できます。ユーザーブラウザから W&B アプリ UI へのトラフィックで同じ仕組みを利用したい場合は、コーポレートネットワークからクラウドアカウント内のプライベートエンドポイントへの適切な DNS ベースのルーティング設定が必要です。

{{% alert %}}
この機能のご利用をご希望の場合は、お使いの W&B チームまでご連絡ください。
{{% /alert %}}

セキュアなプライベート接続は、[IP 許可リスト]({{< relref path="./ip-allowlisting.md" lang="ja" >}}) と併用できます。IP 許可リストでセキュアなプライベート接続を利用する場合、W&B では AI ワークロードからのトラフィックすべて、また可能であればユーザーブラウザからのトラフィックの大部分にもセキュアなプライベート接続を使用し、特権的な場所からのインスタンス管理には IP 許可リストを利用することを推奨しています。