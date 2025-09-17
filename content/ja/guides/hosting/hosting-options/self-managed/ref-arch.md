---
title: リファレンス アーキテクチャー
description: W&B リファレンス アーキテクチャー
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-ref-arch
    parent: self-managed
weight: 1
---

このページでは、W&B のデプロイメント向けリファレンス アーキテクチャーを説明し、プラットフォームをプロダクション デプロイメントで運用するために推奨されるインフラストラクチャーとリソースを示します。

W&B のデプロイメント 環境の選択に応じて、さまざまなサービスを活用してデプロイメントの堅牢性を高められます。

たとえば、大手 クラウド プロバイダーは堅牢なマネージド データベース サービスを提供しており、データベースの 設定、保守、高可用性、レジリエンスの複雑さを軽減できます。

このリファレンス アーキテクチャーは一般的なデプロイメント シナリオを扱い、最適なパフォーマンスと信頼性のために、クラウド ベンダーのサービスと W&B デプロイメントをどのように統合できるかを示します。

## 始める前に

アプリケーションをプロダクションで運用するには固有の課題が伴い、W&B も例外ではありません。可能な限りプロセスを簡素化していますが、アーキテクチャーや設計上の判断によっては複雑さが生じる場合があります。一般的に、プロダクション デプロイメントの管理には、ハードウェア、オペレーティング システム、ネットワーキング、ストレージ、セキュリティ、W&B プラットフォーム本体、その他の依存関係といった多様なコンポーネントの監督が含まれます。この責任は、初期の 環境 構築から継続的な保守まで及びます。

W&B のセルフマネージド運用が、チームと要件に適しているかを慎重にご検討ください。

プロダクション グレードのアプリケーションを運用・保守するための十分な理解は、セルフマネージドの W&B をデプロイするうえで重要な前提条件です。支援が必要な場合は、当社の Professional Services チームやパートナーが導入と最適化をサポートします。

ご自身で管理せずに W&B を運用できるマネージド ソリューションについては、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) および [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を参照してください。

## インフラストラクチャー

{{< img src="/images/hosting/reference_architecture.png" alt="W&B のインフラストラクチャー図" >}}

### アプリケーション レイヤー

アプリケーション レイヤーは、ノード障害に耐性のあるマルチノードの Kubernetes クラスターで構成されます。Kubernetes クラスターは W&B の Pod を実行・管理します。

### ストレージ レイヤー

ストレージ レイヤーは MySQL データベースとオブジェクト ストレージで構成されます。MySQL データベースはメタデータを、オブジェクト ストレージは モデル や データセット といったアーティファクトを保存します。

## インフラストラクチャー要件

### Kubernetes
W&B Server アプリケーションは、複数の Pod をデプロイする [Kubernetes Operator]({{< relref path="kubernetes-operator/" lang="ja" >}}) としてデプロイされます。このため、W&B には次の要件を満たす Kubernetes クラスターが必要です:
- 完全に 設定 され正しく動作している Ingress コントローラ。
- Persistent Volumes をプロビジョニングできる機能。

### MySQL
W&B はメタデータを MySQL データベースに保存します。データベースのパフォーマンスとストレージ要件は、モデル パラメータや関連メタデータの形状に依存します。例えば、より多くのトレーニング run を追跡するとデータベース サイズは増加し、run テーブル、ユーザー ワークスペース、レポート へのクエリに応じてデータベース負荷も高まります。

セルフマネージドの MySQL データベースをデプロイする際は、以下を考慮してください:

- バックアップ: データベースは定期的に別の施設へバックアップしてください。W&B は保持期間 1 週間以上の毎日バックアップを推奨します。
- パフォーマンス: サーバーが動作しているディスクは高速である必要があります。W&B は SSD または高速化された NAS 上での運用を推奨します。
- 監視: データベースの負荷を監視してください。CPU 使用率が 5 分以上にわたりシステムの 40% 超で張り付く場合、サーバーのリソース不足である可能性が高いサインです。
- 可用性: 可用性と耐久性の要件に応じて、プライマリ サーバーからの更新をリアルタイムでストリーミングし、プライマリがクラッシュまたは破損した際にフェイルオーバーできる、別マシン上のホット スタンバイを構成することを検討してください。

### Redis
W&B は、ジョブ キューイングと データ キャッシュに W&B の各コンポーネントが使用する、単一ノードの Redis 7.x デプロイメントに依存します。PoC のテストや開発を容易にするため、W&B Self-Managed にはローカルの Redis デプロイメントが含まれますが、これはプロダクションには適しません。

W&B は、以下の 環境 にある Redis インスタンスに接続できます:

- [AWS Elasticache](https://aws.amazon.com/pm/elasticache/)
- [Google Cloud Memory Store](https://cloud.google.com/memorystore?hl=en)
- [Azure Cache for Redis](https://azure.microsoft.com/en-us/products/cache)
- お使いの クラウド または オンプレミス のインフラストラクチャーでホストされる Redis デプロイメント

### オブジェクト ストレージ
W&B には、事前署名 URL と CORS をサポートするオブジェクト ストレージが必要です。次のいずれかの形態でデプロイしてください:

- [CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage) は、AI ワークロード向けに最適化された高性能な S3 互換オブジェクト ストレージ サービスです。
- [Amazon S3](https://aws.amazon.com/s3/) は、業界をリードするスケーラビリティ、データ可用性、セキュリティ、パフォーマンスを提供するオブジェクト ストレージ サービスです。
- [Google Cloud Storage](https://cloud.google.com/storage) は、非構造化 データ を大規模に保存するためのマネージド サービスです。
- [Azure Blob Storage](https://azure.microsoft.com/products/storage/blobs) は、テキスト、バイナリ データ、画像、動画、ログなどの大量の非構造化 データを保存するための クラウド ベースのオブジェクト ストレージです。
- [MinIO](https://github.com/minio/minio) などの S3 互換ストレージ（お使いの クラウド または オンプレミス のインフラストラクチャーでホスト）

### バージョン
| ソフトウェア | 最小バージョン                           |
| ------------ | ---------------------------------------- |
| Kubernetes   | v1.29                                    |
| MySQL        | v8.0.0（"General Availability" リリースのみ） |
| Redis        | v7.x                                     |

### ネットワーキング

ネットワーク化されたデプロイメントでは、インストール と 実行時 の _両方_ に、以下のエンドポイントへの egress が必要です:
* https://deploy.wandb.ai
* https://charts.wandb.ai
* https://docker.io
* https://quay.io
* `https://gcr.io`

エアギャップ環境でのデプロイメントについては、[Kubernetes operator for air-gapped instances]({{< relref path="kubernetes-operator/operator-airgapped.md" lang="ja" >}}) を参照してください。
トレーニング用インフラおよび 実験 を追跡する各システムから、W&B とオブジェクト ストレージへの アクセス が必要です。

### DNS
W&B デプロイメントの FQDN（完全修飾ドメイン名）は、A レコードを用いて Ingress / ロードバランサーの IP アドレスに解決できる必要があります。

### SSL/TLS
W&B は、クライアントと サーバー 間の通信を保護するために有効な署名付き SSL/TLS 証明書を必要とします。SSL/TLS の終端は Ingress / ロードバランサーで行う必要があります。W&B Server アプリケーションは SSL/TLS 接続を終端しません。

注意: 自己署名証明書や独自 CA の使用は推奨しません。

### サポートされている CPU アーキテクチャー
W&B は Intel（x86）CPU アーキテクチャー上で動作します。ARM はサポートしていません。

## インフラストラクチャーのプロビジョニング
プロダクション向けのデプロイには Terraform の利用を推奨します。Terraform では、必要なリソース、それらの相互参照、依存関係を定義できます。W&B は主要な クラウド プロバイダー向けの Terraform モジュールを提供しています。詳細は、[セルフマネージドのクラウド アカウント内に W&B Server をデプロイする]({{< relref path="/guides/hosting/hosting-options/self-managed.md#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) を参照してください。

## サイジング
以下の一般的なガイドラインをデプロイ計画の出発点としてご利用ください。新規デプロイメントではすべてのコンポーネントを継続的に監視し、観測された利用状況に基づいて調整することを推奨します。プロダクション デプロイメントでも、最適なパフォーマンスを維持できるよう継続的に監視し、必要に応じて調整してください。

### Models のみ

#### Kubernetes

| 環境          | CPU      | メモリ  | ディスク |
| ------------- | -------- | ------- | -------- |
| テスト/開発   | 2 cores  | 16 GB   | 100 GB   |
| プロダクション | 8 cores  | 64 GB   | 100 GB   |

数値は Kubernetes ワーカーノード 1 台あたりです。

#### MySQL

| 環境          | CPU      | メモリ  | ディスク |
| ------------- | -------- | ------- | -------- |
| テスト/開発   | 2 cores  | 16 GB   | 100 GB   |
| プロダクション | 8 cores  | 64 GB   | 500 GB   |

数値は MySQL ノード 1 台あたりです。

### Weave のみ

#### Kubernetes

| 環境          | CPU       | メモリ | ディスク |
| ------------- | --------- | ------ | -------- |
| テスト/開発   | 4 cores   | 32 GB  | 100 GB   |
| プロダクション | 12 cores  | 96 GB  | 100 GB   |

数値は Kubernetes ワーカーノード 1 台あたりです。

#### MySQL

| 環境          | CPU      | メモリ  | ディスク |
| ------------- | -------- | ------- | -------- |
| テスト/開発   | 2 cores  | 16 GB   | 100 GB   |
| プロダクション | 8 cores  | 64 GB   | 500 GB   |

数値は MySQL ノード 1 台あたりです。

### Models と Weave

#### Kubernetes

| 環境          | CPU       | メモリ   | ディスク |
| ------------- | --------- | -------- | -------- |
| テスト/開発   | 4 cores   | 32 GB    | 100 GB   |
| プロダクション | 16 cores  | 128 GB   | 100 GB   |

数値は Kubernetes ワーカーノード 1 台あたりです。

#### MySQL

| 環境          | CPU      | メモリ  | ディスク |
| ------------- | -------- | ------- | -------- |
| テスト/開発   | 2 cores  | 16 GB   | 100 GB   |
| プロダクション | 8 cores  | 64 GB   | 500 GB   |

数値は MySQL ノード 1 台あたりです。

## クラウド プロバイダーのインスタンス推奨

### サービス

| クラウド | Kubernetes | MySQL                   | オブジェクト ストレージ        |
| -------- | ---------- | ----------------------- | ------------------------------- |
| AWS      | EKS        | RDS Aurora              | S3                              |
| GCP      | GKE        | Google Cloud SQL - Mysql | Google Cloud Storage (GCS)      |
| Azure    | AKS        | Azure Database for Mysql | Azure Blob Storage              |

### マシンタイプ

これらの推奨は、クラウド インフラストラクチャーにおける W&B のセルフマネージド デプロイメントの各ノードに適用されます。

#### AWS

| 環境        | K8s（Models のみ） | K8s（Weave のみ） | K8s（Models & Weave） | MySQL          |
| ----------- | ------------------ | ----------------- | --------------------- | -------------- |
| テスト/開発 | r6i.large          | r6i.xlarge        | r6i.xlarge            | db.r6g.large   |
| プロダクション | r6i.2xlarge        | r6i.4xlarge       | r6i.4xlarge           | db.r6g.2xlarge |

#### GCP

| 環境        | K8s（Models のみ） | K8s（Weave のみ） | K8s（Models & Weave） | MySQL           |
| ----------- | ------------------ | ----------------- | --------------------- | --------------- |
| テスト/開発 | n2-highmem-2       | n2-highmem-4      | n2-highmem-4          | db-n1-highmem-2 |
| プロダクション | n2-highmem-8       | n2-highmem-16     | n2-highmem-16         | db-n1-highmem-8 |

#### Azure

| 環境        | K8s（Models のみ） | K8s（Weave のみ） | K8s（Models & Weave） | MySQL               |
| ----------- | ------------------ | ----------------- | --------------------- | ------------------- |
| テスト/開発 | Standard_E2_v5     | Standard_E4_v5    | Standard_E4_v5        | MO_Standard_E2ds_v4 |
| プロダクション | Standard_E8_v5     | Standard_E16_v5   | Standard_E16_v5       | MO_Standard_E8ds_v4 |