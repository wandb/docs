---
title: Configure IP allowlisting for Dedicated Cloud
menu:
  default:
    identifier: ja-guides-hosting-data-security-ip-allowlisting
    parent: data-security
weight: 3
---

[ 専用クラウド ]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンスへの アクセス を、許可された IP アドレス のリストのみに制限できます。これは、AI ワークロードから W&B API への アクセス と、 ユーザー のブラウザーから W&B アプリ UI への アクセス の両方に適用されます。 専用クラウド インスタンスに対して IP アドレス許可リストが設定されると、W&B は許可されていない場所からのリクエストをすべて拒否します。 専用クラウド インスタンスの IP アドレス許可リストを構成するには、W&B チームにお問い合わせください。

IP アドレス許可リストは、AWS、GCP、Azure の 専用クラウド インスタンスで利用できます。

IP アドレス許可リストは、[セキュアプライベート接続]({{< relref path="./private-connectivity.md" lang="ja" >}}) で使用できます。セキュアプライベート接続で IP アドレス許可リストを使用する場合、W&B は、AI ワークロードからのすべてのトラフィックと、可能な場合は ユーザー ブラウザーからのトラフィックの大部分にセキュアプライベート接続を使用し、特権のある場所からのインスタンス管理には IP アドレス許可リストを使用することをお勧めします。

{{% alert color="secondary" %}}
W&B では、個々の「/32」IP アドレスではなく、企業またはビジネスの出力ゲートウェイに割り当てられた [CIDR ブロック](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) を使用することを強くお勧めします。個々の IP アドレスの使用はスケーラブルではなく、クラウドごとに厳密な制限があります。
{{% /alert %}}
