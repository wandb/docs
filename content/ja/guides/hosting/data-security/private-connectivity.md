---
title: Configure private connectivity to Dedicated Cloud
menu:
  default:
    identifier: ja-guides-hosting-data-security-private-connectivity
    parent: data-security
weight: 4
---

[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) インスタンスに、クラウドプロバイダのセキュアなプライベートネットワークを介して接続することができます。これは、AI ワークロードから W&B API へのアクセス、およびオプションでユーザーブラウザから W&B アプリ UI へのアクセスにも適用されます。プライベート接続を使用する場合、関連するリクエストとレスポンスはパブリックネットワークやインターネットを通じて送信されません。

{{% alert %}}
セキュアなプライベート接続は、専用クラウドの高度なセキュリティオプションとしてまもなく利用可能になります。
{{% /alert %}}

セキュアなプライベート接続は、AWS、GCP、Azure 上の専用クラウドインスタンスで利用可能です：

* AWS で [AWS Privatelink](https://aws.amazon.com/privatelink/) を使用
* GCP で [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) を使用
* Azure で [Azure Private Link](https://azure.microsoft.com/en-us/products/private-link) を使用

一度有効にすると、W&B はインスタンスのプライベートエンドポイントサービスを作成し、関連する DNS URI を提供します。それにより、クラウドアカウント内でプライベートエンドポイントを作成し、関連するトラフィックをプライベートエンドポイントサービスにルーティングできます。プライベートエンドポイントは、クラウド VPC や VNet 内で実行される AI トレーニングワークロードに対して簡単にセットアップできます。ユーザーブラウザから W&B アプリ UI へのトラフィックにも同じメカニズムを使用するには、企業ネットワークからクラウドアカウント内のプライベートエンドポイントへの適切な DNS ベースのルーティングを設定する必要があります。

{{% alert %}}
この機能を使用したい場合は、W&B チームに連絡してください。
{{% /alert %}}

[IP allowlisting]({{< relref path="./ip-allowlisting.md" lang="ja" >}}) とセキュアなプライベート接続を使用できます。IP allowlisting にセキュアなプライベート接続を使用する場合、W&B は、AI ワークロードからのすべてのトラフィックと、可能であればユーザーブラウザからの大部分のトラフィックにセキュアなプライベート接続を使用し、特権のある場所からのインスタンス管理には IP allowlisting を使用することを推奨します。