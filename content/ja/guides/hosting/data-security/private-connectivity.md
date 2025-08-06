---
title: 専用クラウドへのプライベート接続を設定する
menu:
  default:
    identifier: private-connectivity
    parent: data-security
weight: 4
---

[専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) インスタンスには、クラウドプロバイダーのセキュアなプライベートネットワーク経由で接続できます。これは、AI ワークロードから W&B API へのアクセス、および必要に応じてユーザーのブラウザから W&B アプリ UI へのアクセスにも適用されます。プライベート接続を利用する場合、該当するリクエストやレスポンスは、パブリックネットワークやインターネットを経由しません。

{{% alert %}}
セキュアなプライベート接続は、専用クラウドの高度なセキュリティオプションとして近日提供予定です。
{{% /alert %}}

セキュアなプライベート接続は、AWS、GCP、Azure 上の専用クラウドインスタンスで利用可能です：

* AWS では [AWS Privatelink](https://aws.amazon.com/privatelink/) を利用
* GCP では [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) を利用
* Azure では [Azure Private Link](https://azure.microsoft.com/products/private-link) を利用

有効化すると、W&B がインスタンス用のプライベートエンドポイントサービスを作成し、接続用の該当 DNS URI を提供します。これにより、クラウドアカウント内でプライベートエンドポイントを作成し、関連トラフィックをプライベートエンドポイントサービスにルーティングできます。プライベートエンドポイントは、クラウドの VPC や VNet 内で実行されている AI トレーニングワークロードにとって、設定が簡単です。ユーザーのブラウザから W&B アプリ UI へのトラフィックにも同じ仕組みを利用する場合は、社内ネットワークからクラウドアカウント内プライベートエンドポイントへの DNS ベースのルーティングを適切に設定する必要があります。

{{% alert %}}
この機能の利用をご希望の場合は、W&B チームまでご連絡ください。
{{% /alert %}}

[IP 許可リスト]({{< relref "./ip-allowlisting.md" >}}) と併用して、セキュアなプライベート接続を利用できます。IP 許可リストにセキュアなプライベート接続を使う場合、AI ワークロードからの全トラフィックと、可能であればユーザーのブラウザからの大部分のトラフィックについてもセキュアなプライベート接続を利用し、特権的な場所からインスタンス管理を行う用途でのみ IP 許可リストを利用することを W&B では推奨しています。