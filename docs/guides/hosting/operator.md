---
description: Kubernetes Operator を使用した W&B サーバーのホスティング
displayed_sidebar: default
---


# W&B Kubernetes Operator

W&B Kubernetes Operator を使用すると、W&B Server のデプロイメント、管理、トラブルシューティング、およびスケーリングを簡素化できます。このオペレーターは、W&B インスタンスのスマートアシスタントのように考えることができます。

W&B Server のアーキテクチャーとデザインは、ユーザー向けの AI 開発ツール機能を拡張し、高パフォーマンス、より良いスケーラビリティ、および簡単な管理のための適切なプリミティブを持つように、継続的に進化しています。この進化は、コンピューティングサービス、関連するストレージ、およびそれらの間の接続にも適用されます。W&B は、このオペレーターを使用して、デプロイメントの種類に関係なく、ユーザーにこれらの改善を展開することを計画しています。

:::info
W&B は、AWS、GCP、Azure のパブリッククラウド上の専用クラウドインスタンスをデプロイおよび管理するためにオペレーターを使用しています。
:::

オペレーターに関する一般的な情報については、Kubernetes ドキュメントの [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) を参照してください。

## セルフマネージドインスタンスを W&B Kubernetes Operator に移行する
W&B Server インスタンスをセルフマネージしている場合は、オペレーターの使用をお勧めします。これにより、W&B は新しいサービスや製品をインスタンスにシームレスに展開し、より良いトラブルシューティングとサポートを提供できます。

:::note
セルフマネージド W&B Server デプロイメント用のオペレーターはプライベートプレビュー中です。GA 時またはその後には、オペレーターを使用しないデプロイメントメカニズムを廃止する予定です。質問がある場合は、[Customer Support](mailto:support@wandb.com) または W&B チームにお問い合わせください。
:::

## クラウド Terraform モジュールに予め組み込まれている

W&B Kubernetes Operator は、以下のバージョンで公式の W&B クラウド固有の Terraform Modules に予め組み込まれています:

| Terraform Module                                 | Version |
| ------------------------------------------------ | ------- |
| https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

このインテグレーションにより、W&B Kubernetes Operator は最小限の設定でインスタンスで使用できるようになり、クラウド環境内での W&B Server のデプロイおよび管理のための円滑なパスを提供します。

## Helm Terraform モジュールを使用したデプロイ

公式の W&B Terraform Module [terraform-helm-wandb](https://github.com/wandb/terraform-helm-wandb) を使用して W&B Kubernetes Operator をインストールします。

この方法は、特定の要件に合わせたカスタマイズされたデプロイメントを可能にし、一貫性と再現性のために Terraform のインフラストラクチャー・アズ・コードのアプローチを活用します。

:::note
オペレーターの使用方法の詳細については、[Customer Support](mailto:support@wandb.com) または W&B チームにお問い合わせください。
:::
