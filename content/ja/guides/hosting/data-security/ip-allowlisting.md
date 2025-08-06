---
title: 専用クラウドでの IP 許可リストを設定する
menu:
  default:
    identifier: ip-allowlisting
    parent: data-security
weight: 3
---

[専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) インスタンスへのアクセスを、許可された IP アドレスのリストのみに制限することができます。これは、AI ワークロードから W&B API へのアクセス、およびユーザーのブラウザから W&B アプリ UI へのアクセスの両方に適用されます。専用クラウドインスタンスで IP アローリストが設定されると、W&B は未承認の場所からのリクエストをすべて拒否します。専用クラウドインスタンスの IP アローリスティング設定については、W&B チームまでご連絡ください。

IP アローリスティングは、AWS、GCP、Azure 上の専用クラウドインスタンスでご利用いただけます。

[セキュアなプライベート接続]({{< relref "./private-connectivity.md" >}}) と併用することも可能です。IP アローリスティングをセキュアなプライベート接続と組み合わせる場合、AI ワークロードからのすべてのトラフィックおよび可能な限りユーザーブラウザからの大部分のトラフィックにはセキュアなプライベート接続を利用し、特権のある場所からのインスタンス管理には IP アローリスティングを利用することを W&B では推奨しています。

{{% alert color="secondary" %}}
W&B では、個別の `/32` IP アドレスではなく、企業や組織のエグレスゲートウェイに割り当てられている [CIDR ブロック](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) の利用を強く推奨しています。個別の IP アドレスは拡張性がなく、クラウドごとに厳しい制限があります。
{{% /alert %}}