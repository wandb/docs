---
title: 専用クラウドへのプライベート接続を設定する
menu:
  default:
    identifier: ja-guides-hosting-data-security-private-connectivity
    parent: data-security
weight: 4
---

クラウド プロバイダーのセキュアなプライベートネットワークを介して、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) インスタンスに接続できます。これは、AI ワークロードから W&B API へのアクセスに加え、オプションでユーザー ブラウザから W&B アプリ UI へのアクセスにも適用されます。プライベート接続を使用する場合、関連するリクエストとレスポンスはパブリックネットワークやインターネットを経由しません。

{{% alert %}}
セキュアなプライベート接続は、専用クラウドの高度なセキュリティ オプションとして近日中に提供開始予定です。
{{% /alert %}}

セキュアなプライベート接続は、AWS、GCP、Azure の専用クラウド インスタンスで利用できます。

* AWS では [AWS PrivateLink](https://aws.amazon.com/privatelink/) を使用
* GCP では [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) を使用
* Azure では [Azure Private Link](https://azure.com/products/private-link) を使用


有効化されると、W&B はインスタンス用のプライベートエンドポイントサービスを作成し、接続に必要な DNS URI を提供します。これにより、クラウド アカウント内にプライベートエンドポイントを作成し、関連するトラフィックをプライベートエンドポイントサービスにルーティングできます。プライベートエンドポイントは、クラウド VPC または VNet 内で実行される AI トレーニング ワークロード向けに、より簡単に設定できます。ユーザー ブラウザから W&B アプリ UI へのトラフィックにも同じメカニズムを使う場合は、社内ネットワークからクラウド アカウント内のプライベートエンドポイントへ向けて、適切な DNS ベースのルーティングを設定する必要があります。

{{% alert %}}
この機能の使用をご希望の場合は、W&B チームにお問い合わせください。
{{% /alert %}}

セキュアなプライベート接続は、[IP 許可リスト]({{< relref path="./ip-allowlisting.md" lang="ja" >}}) と併用できます。IP 許可リストとセキュアなプライベート接続を併用する場合、W&B では、特権のある場所からのインスタンス管理には IP 許可リストを使い、可能な限り AI ワークロード発の全トラフィックとユーザー ブラウザ発トラフィックの大部分にはセキュアなプライベート接続を使用することを推奨します。