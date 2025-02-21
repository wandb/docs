---
title: Configure private connectivity to Dedicated Cloud
menu:
  default:
    identifier: ja-guides-hosting-data-security-private-connectivity
    parent: data-security
weight: 4
---

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) インスタンスには、クラウド プロバイダーのセキュアなプライベートネットワークを介して接続できます。これは、AI ワークロードから W&B API へのアクセス、およびオプションで ユーザー のブラウザから W&B アプリ UI へのアクセスにも適用されます。プライベート接続を使用する場合、関連するリクエストとレスポンスは、パブリックネットワークまたはインターネットを経由しません。

{{% alert %}}
セキュアなプライベート接続は、高度なセキュリティオプションとして 専用クラウド で近日提供予定です。
{{% /alert %}}

セキュアなプライベート接続は、AWS、GCP、Azure の 専用クラウド インスタンスで利用できます。

* AWS での [AWS Privatelink](https://aws.amazon.com/privatelink/) の使用
* GCP での [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) の使用
* Azure での [Azure Private Link](https://azure.microsoft.com/en-us/products/private-link) の使用

有効にすると、W&B はインスタンスのプライベートエンドポイントサービスを作成し、接続するための関連する DNS URI を提供します。これにより、クラウド アカウントにプライベートエンドポイントを作成して、関連するトラフィックをプライベートエンドポイントサービスにルーティングできます。プライベートエンドポイントは、クラウド VPC または VNet 内で実行されている AI トレーニング ワークロード用に簡単にセットアップできます。 ユーザー のブラウザから W&B アプリ UI へのトラフィックに同じメカニズムを使用するには、企業ネットワークからクラウド アカウントのプライベートエンドポイントへの適切な DNS ベースのルーティングを構成する必要があります。

{{% alert %}}
この機能を使用したい場合は、W&B チームにお問い合わせください。
{{% /alert %}}

セキュアなプライベート接続は、[IP許可リスト]({{< relref path="./ip-allowlisting.md" lang="ja" >}}) で使用できます。IP許可リストにセキュアなプライベート接続を使用する場合、W&B は、AI ワークロードからのすべてのトラフィックと、可能な場合は ユーザー のブラウザからのトラフィックの大部分に対してセキュアなプライベート接続を保護し、特権のある場所からのインスタンス管理に IP 許可リストを使用することをお勧めします。
