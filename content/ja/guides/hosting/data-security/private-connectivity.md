---
title: 専用クラウドへのプライベート接続を設定します
menu:
  default:
    identifier: ja-guides-hosting-data-security-private-connectivity
    parent: data-security
weight: 4
---

クラウドプロバイダーの安全なプライベートネットワークを介して、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) インスタンスに接続できます。これは、AI ワークロードから W&B API へのアクセス、およびオプションでユーザーのブラウザから W&B アプリ UI へのアクセスにも適用されます。プライベート接続を使用する場合、関連するリクエストとレスポンスはパブリックネットワークやインターネットを経由しません。

{{% alert %}}
安全なプライベート接続は、専用クラウドの高度なセキュリティオプションとして間もなく利用可能になります。
{{% /alert %}}

安全なプライベート接続は、AWS、GCP、および Azure 上の専用クラウドインスタンスで利用可能です：

* AWS で [AWS Privatelink](https://aws.amazon.com/privatelink/) を使用
* GCP で [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) を使用
* Azure で [Azure Private Link](https://learn.microsoft.com/en-us/azure/private-link/private-link-overview) を使用

一度有効にすると、W&B はインスタンス用のプライベートエンドポイントサービスを作成し、接続するための関連する DNS URI を提供します。それにより、クラウドアカウント内にプライベートエンドポイントを作成し、関連するトラフィックをプライベートエンドポイントサービスにルーティングできます。プライベートエンドポイントは、クラウド VPC または VNet 内で動作する AI トレーニングワークロードに対して、設定が容易です。ユーザーブラウザから W&B アプリ UI へのトラフィックに対しても同じメカニズムを使用するには、企業ネットワークからクラウドアカウント内のプライベートエンドポイントへの適切な DNS ベースのルーティングを設定する必要があります。

{{% alert %}}
この機能を使用したい場合は、W&B チームにご連絡ください。
{{% /alert %}}

[IP allowlisting]({{< relref path="./ip-allowlisting.md" lang="ja" >}}) とともに安全なプライベート接続を使用できます。IP allowlisting のために安全なプライベート接続を使用する場合、W&B は可能であれば、IP allowlisting を特権的な場所からのインスタンス管理のために使用しつつ、AI ワークロードからのすべてのトラフィックと、ユーザーブラウザからのトラフィックの大部分に対して安全なプライベート接続を確保することをお勧めします。