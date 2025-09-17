---
title: 専用クラウド向け IP allowlisting の設定
menu:
  default:
    identifier: ja-guides-hosting-data-security-ip-allowlisting
    parent: data-security
weight: 3
---

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンスへのアクセスは、許可された IP アドレスのリストからのみに制限できます。これは、AI ワークロードから W&B API へのアクセス、およびユーザーのブラウザから W&B アプリ UI へのアクセスにも適用されます。専用クラウドインスタンスに IP 許可リストが設定されると、W&B は許可されていないロケーションからのすべてのリクエストを拒否します。専用クラウドインスタンスの IP 許可リストを設定するには、W&B チームにご連絡ください。

IP 許可リストは、AWS、GCP、および Azure 上の専用クラウドインスタンスで利用できます。

IP 許可リストは、[セキュアなプライベート接続]({{< relref path="./private-connectivity.md" lang="ja" >}}) とともに使用できます。セキュアなプライベート接続と IP 許可リストを併用する場合、W&B は、AI ワークロードからのすべてのトラフィックと、可能であればユーザーのブラウザからのトラフィックの大部分にはセキュアなプライベート接続を使用し、許可されたロケーションからのインスタンス管理には IP 許可リストを使用することを推奨します。

{{% alert color="secondary" %}}
W&B は、個別の `/32` IP アドレスではなく、企業や組織の Egress ゲートウェイに割り当てられた [CIDR ブロック](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) を使用することを強く推奨します。個別の IP アドレスの使用はスケーラブルではなく、クラウドごとに厳格な制限があります。
{{% /alert %}}