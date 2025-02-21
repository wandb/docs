---
title: Reference Architecture
description: W&B リファレンス アーキテクチャー
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-ref-arch
    parent: self-managed
weight: 1
---

このページでは、Weights & Biases のデプロイメントのリファレンスアーキテクチャについて説明し、プラットフォームのプロダクションデプロイメントをサポートするために推奨されるインフラストラクチャとリソースの概要を示します。

Weights & Biases（W&B）に選択したデプロイメント環境に応じて、デプロイメントの回復性を高めるのに役立つさまざまなサービスがあります。

たとえば、主要なクラウドプロバイダーは、データベースの構成、メンテナンス、高可用性、および回復性の複雑さを軽減するのに役立つ、堅牢なマネージドデータベースサービスを提供しています。

このリファレンスアーキテクチャは、一般的なデプロイメントシナリオに対応し、最適なパフォーマンスと信頼性を実現するために、W&B のデプロイメントをクラウドベンダーのサービスと統合する方法を示します。

## 開始する前に

プロダクション環境でアプリケーションを実行するには、固有の一連の課題があり、W&B も例外ではありません。プロセスの合理化を目指していますが、固有のアーキテクチャと設計上の決定によっては、特定の複雑さが発生する可能性があります。通常、プロダクションデプロイメントの管理には、ハードウェア、オペレーティングシステム、ネットワーキング、ストレージ、セキュリティ、W&B プラットフォーム自体、およびその他の依存関係を含む、さまざまなコンポーネントの監視が含まれます。この責任は、環境の初期セットアップとその継続的なメンテナンスの両方に及びます。

W&B を使用した自己管理アプローチが、チームと特定の要件に適しているかどうかを慎重に検討してください。

プロダクショングレードのアプリケーションを実行および保守する方法をしっかりと理解しておくことは、自己管理型の W&B をデプロイする上で重要な前提条件です。チームが支援を必要とする場合、当社のプロフェッショナルサービスチームとパートナーが、実装と最適化のサポートを提供します。

W&B の実行を自身で管理するのではなく、マネージドソリューションの詳細については、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) および [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を参照してください。

## インフラストラクチャ

{{< img src="/images/hosting/reference_architecture.png" alt="W&B infrastructure diagram" >}}

### アプリケーション層

アプリケーション層は、ノード障害に対する回復性を持つマルチノード Kubernetes クラスターで構成されています。Kubernetes クラスターは、W&B のポッドを実行および維持します。

### ストレージ層

ストレージ層は、MySQL データベースとオブジェクトストレージで構成されています。MySQL データベースはメタデータを格納し、オブジェクトストレージはモデルやデータセットなどの Artifacts を格納します。

## インフラストラクチャ要件

### Kubernetes
W&B Server アプリケーションは、複数のポッドをデプロイする [Kubernetes Operator]({{< relref path="kubernetes-operator/" lang="ja" >}}) としてデプロイされます。このため、W&B には次のものを持つ Kubernetes クラスターが必要です。
- 完全に構成され、機能する Ingress コントローラ。
- Persistent Volumes をプロビジョニングする機能。

### MySQL
W&B は、メタデータを MySQL データベースに格納します。データベースのパフォーマンスとストレージ要件は、モデルパラメータと関連メタデータの形状によって異なります。たとえば、より多くの training runs を追跡するにつれてデータベースのサイズは大きくなり、run tables、ユーザー Workspace 、 Reports でのクエリに基づいてデータベースへの負荷が増加します。

自己管理型の MySQL データベースをデプロイする際は、以下を検討してください。

- **バックアップ**。データベースを定期的に別の施設にバックアップする必要があります。W&B は、少なくとも 1 週間の保持期間で毎日のバックアップを推奨します。
- **パフォーマンス**。サーバーが実行されているディスクは高速である必要があります。W&B は、SSD または高速化された NAS でデータベースを実行することを推奨します。
- **監視**。データベースの負荷を監視する必要があります。CPU 使用率がシステムの 40% を超えて 5 分以上継続する場合は、サーバーのリソースが不足している可能性が高いことを示しています。
- **可用性**。可用性と耐久性の要件に応じて、プライマリサーバーからのすべての更新をリアルタイムでストリーミングし、プライマリサーバーがクラッシュまたは破損した場合にフェイルオーバーに使用できる、別のマシン上にホットスタンバイを構成することができます。

### オブジェクトストレージ
W&B には、次のいずれかにデプロイされた、事前署名付き URL と CORS をサポートするオブジェクトストレージが必要です。
- Amazon S3
- Azure Cloud Storage
- Google Cloud Storage
- Amazon S3 と互換性のあるストレージサービス

### バージョン
| ソフトウェア | 最小バージョン |
| --- | --- |
| Kubernetes | v1.29 |
| MySQL | v8.0.0、「一般公開」リリースのみ |

### ネットワーク

ネットワーク化されたデプロイメントの場合、_インストール_時とランタイム時に、これらのエンドポイントへの出力が必要です。
* https://deploy.wandb.ai
* https://charts.wandb.ai
* https://docker.io
* https://quay.io
* `https://gcr.io`

エアギャップされたデプロイメントの詳細については、[エアギャップされたインスタンスの Kubernetes operator]({{< relref path="kubernetes-operator/operator-airgapped.md" lang="ja" >}}) を参照してください。
training インフラストラクチャと、 Experiments のニーズを追跡する各システムには、W&B とオブジェクトストレージへのアクセスが必要です。

### DNS
W&B デプロイメントの完全修飾ドメイン名（FQDN）は、A レコードを使用してイングレス/ロードバランサーの IP アドレスに解決される必要があります。

### SSL/TLS
W&B には、クライアントとサーバー間の安全な通信のための有効な署名付き SSL/TLS 証明書が必要です。SSL/TLS 終端は、イングレス/ロードバランサーで発生する必要があります。W&B Server アプリケーションは、SSL または TLS 接続を終端しません。

注意：W&B は、自己署名証明書とカスタム CA の使用を推奨していません。

### サポートされている CPU アーキテクチャ
W&B は、Intel（x86）CPU アーキテクチャで実行されます。ARM はサポートされていません。

## インフラストラクチャのプロビジョニング
Terraform は、本番環境向けに W&B をデプロイするための推奨される方法です。Terraform を使用すると、必要なリソース、他のリソースへの参照、およびそれらの依存関係を定義できます。W&B は、主要なクラウドプロバイダー向けの Terraform モジュールを提供しています。詳細については、[自己管理型クラウドアカウント内での W&B Server のデプロイ]({{< relref path="/guides/hosting/hosting-options/self-managed.md#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) を参照してください。

## サイジング
デプロイメントを計画する際の出発点として、次の一般的なガイドラインを使用してください。W&B は、新しいデプロイメントのすべてのコンポーネントを注意深く監視し、観察された使用パターンに基づいて調整することを推奨しています。長期にわたってプロダクションデプロイメントを監視し続け、最適なパフォーマンスを維持するために必要に応じて調整を行ってください。

### Models のみ

#### Kubernetes

| 環境 | CPU | メモリ | ディスク |
| --- | --- | --- | --- |
| テスト/開発 | 2 コア | 16 GB | 100 GB |
| プロダクション | 8 コア | 64 GB | 100 GB |

数値は Kubernetes ワーカーノードあたりです。

#### MySQL

| 環境 | CPU | メモリ | ディスク |
| --- | --- | --- | --- |
| テスト/開発 | 2 コア | 16 GB | 100 GB |
| プロダクション | 8 コア | 64 GB | 500 GB |

数値は MySQL ノードあたりです。

### Weave のみ

#### Kubernetes

| 環境 | CPU | メモリ | ディスク |
| --- | --- | --- | --- |
| テスト/開発 | 4 コア | 32 GB | 100 GB |
| プロダクション | 12 コア | 96 GB | 100 GB |

数値は Kubernetes ワーカーノードあたりです。

#### MySQL

| 環境 | CPU | メモリ | ディスク |
| --- | --- | --- | --- |
| テスト/開発 | 2 コア | 16 GB | 100 GB |
| プロダクション | 8 コア | 64 GB | 500 GB |

数値は MySQL ノードあたりです。

### Models と Weave

#### Kubernetes

| 環境 | CPU | メモリ | ディスク |
| --- | --- | --- | --- |
| テスト/開発 | 4 コア | 32 GB | 100 GB |
| プロダクション | 16 コア | 128 GB | 100 GB |

数値は Kubernetes ワーカーノードあたりです。

#### MySQL

| 環境 | CPU | メモリ | ディスク |
| --- | --- | --- | --- |
| テスト/開発 | 2 コア | 16 GB | 100 GB |
| プロダクション | 8 コア | 64 GB | 500 GB |

数値は MySQL ノードあたりです。

## クラウドプロバイダーのインスタンス推奨事項

### サービス

| クラウド | Kubernetes | MySQL | オブジェクトストレージ |
| --- | --- | --- | --- |
| AWS | EKS | RDS Aurora | S3 |
| GCP | GKE | Google Cloud SQL - Mysql | Google Cloud Storage (GCS) |
| Azure | AKS | Azure Database for Mysql | Azure Blob Storage |

### マシンタイプ

これらの推奨事項は、クラウドインフラストラクチャでの W&B の自己管理型デプロイメントの各ノードに適用されます。

#### AWS

| 環境 | K8s (Models のみ) | K8s (Weave のみ) | K8s (Models&Weave) | MySQL |
| --- | --- | --- | --- | --- |
| テスト/開発 | r6i.large | r6i.xlarge | r6i.xlarge | db.r6g.large |
| プロダクション | r6i.2xlarge | r6i.4xlarge | r6i.4xlarge | db.r6g.2xlarge |

#### GCP

| 環境 | K8s (Models のみ) | K8s (Weave のみ) | K8s (Models&Weave) | MySQL |
| --- | --- | --- | --- | --- |
| テスト/開発 | n2-highmem-2 | n2-highmem-4 | n2-highmem-4 | db-n1-highmem-2 |
| プロダクション | n2-highmem-8 | n2-highmem-16 | n2-highmem-16 | db-n1-highmem-8 |

#### Azure

| 環境 | K8s (Models のみ) | K8s (Weave のみ) | K8s (Models&Weave) | MySQL |
| --- | --- | --- | --- | --- |
| テスト/開発 | Standard_E2_v5 | Standard_E4_v5 | Standard_E4_v5 | MO_Standard_E2ds_v4 |
| プロダクション | Standard_E8_v5 | Standard_E16_v5 | Standard_E16_v5 | MO_Standard_E8ds_v4 |
