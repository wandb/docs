---
title: Configure IP allowlisting for Dedicated Cloud
menu:
  default:
    identifier: ja-guides-hosting-data-security-ip-allowlisting
    parent: data-security
weight: 3
---

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンスへの アクセス を、許可された IP アドレス のリストのみに制限できます。これは、AI ワークロード から W&B API への アクセス と、 ユーザー のブラウザ から W&B アプリ UI への アクセス の両方に適用されます。専用クラウド インスタンスに対して IP アドレス の許可リストが設定されると、W&B は許可されていない場所からのリクエストを拒否します。専用クラウド インスタンスの IP アドレス 許可リストを設定するには、W&B チームにお問い合わせください。

IP アドレス の許可リストは、AWS、GCP、Azure の専用クラウド インスタンスで利用できます。

IP アドレス の許可リストは、[セキュアなプライベート接続]({{< relref path="./private-connectivity.md" lang="ja" >}}) と組み合わせて使用できます。セキュアなプライベート接続で IP アドレス 許可リストを使用する場合、AI ワークロード からのすべてのトラフィックと、可能な場合は ユーザー のブラウザ からのトラフィックの大部分にセキュアなプライベート接続を使用し、特権のある場所からのインスタンス管理には IP アドレス 許可リストを使用することを W&B は推奨します。

{{% alert color="secondary" %}}
W&B は、個々の `/32` IP アドレス ではなく、企業またはビジネスの出口ゲートウェイに割り当てられた [CIDR ブロック](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) を使用することを強く推奨します。個々の IP アドレス の使用は拡張性が低く、クラウド ごとに厳格な制限があります。
{{% /alert %}}
