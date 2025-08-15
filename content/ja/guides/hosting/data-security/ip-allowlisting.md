---
title: 専用クラウドの IP 許可リストを設定する
menu:
  default:
    identifier: ja-guides-hosting-data-security-ip-allowlisting
    parent: data-security
weight: 3
---

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンスへのアクセスを、許可された IP アドレスリストのみに制限することができます。これは、AI ワークロードから W&B API へのアクセスや、ユーザーのブラウザから W&B アプリ UI へのアクセスにも適用されます。IP 許可リストが専用クラウドインスタンスに設定されると、W&B は他の未承認の場所からのリクエストを拒否します。専用クラウドインスタンスで IP 許可リスト設定をご希望の場合は、W&B チームまでご連絡ください。

IP 許可リストは、AWS、GCP、Azure 上の専用クラウドインスタンスでご利用いただけます。

[セキュアなプライベート接続]({{< relref path="./private-connectivity.md" lang="ja" >}}) とあわせて IP 許可リストをご利用いただけます。IP 許可リストとセキュアなプライベート接続を組み合わせる場合は、AI ワークロードからの全トラフィックおよび可能な限りユーザーブラウザからの大部分のトラフィックにセキュアなプライベート接続を利用し、IP 許可リストは管理権限のある場所からのインスタンス管理用に利用することを W&B は推奨します。

{{% alert color="secondary" %}}
W&B は、個別の `/32` IP アドレスではなく、企業やビジネスの出口ゲートウェイに割り当てられた [CIDR ブロック](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) を利用することを強く推奨します。個別の IP アドレスを使用するとスケーラビリティがなく、クラウドごとの厳しい制限があります。
{{% /alert %}}