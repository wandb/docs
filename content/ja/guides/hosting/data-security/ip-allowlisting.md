---
title: Configure IP allowlisting for Dedicated Cloud
menu:
  default:
    identifier: ja-guides-hosting-data-security-ip-allowlisting
    parent: data-security
weight: 3
---

あなたの [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンスへのアクセスを、認証された IP アドレスのリストに限定することができます。これは、AI ワークロードから W&B API へのアクセスや、ユーザーのブラウザから W&B アプリ UI へのアクセスにも適用されます。専用クラウドインスタンスに対して IP アロリスティングを設定すると、それ以外の許可されていない場所からの要求は W&B によって拒否されます。専用クラウドインスタンスの IP アロリスティングを設定するには、W&B チームに連絡してください。

IP アロリスティングは、AWS、GCP、および Azure 上の専用クラウド インスタンスで利用できます。

[安全なプライベート接続]({{< relref path="./private-connectivity.md" lang="ja" >}}) と共に IP アロリスティングを使用できます。もし安全なプライベート接続と共に IP アロリスティングを使用する場合、W&B は AI ワークロードからのすべてのトラフィックと、可能であればユーザーブラウザからのトラフィックの大部分に安全なプライベート接続を使用し、特権を持つ場所からのインスタンス管理に対して IP アロリスティングを使用することを推奨します。

{{% alert color="secondary" %}}
W&B は、個々の `/32` IP アドレスよりも、企業や業務用のエグレスゲートウェイに割り当てられた [CIDR ブロック](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) を使用することを強く推奨します。個々の IP アドレスを使用することはスケーラビリティがなく、クラウドごとに厳しい制限があります。
{{% /alert %}}