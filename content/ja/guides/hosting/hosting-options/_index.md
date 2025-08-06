---
title: デプロイメント オプション
menu:
  default:
    identifier: deployment-options
    parent: w-b-platform
weight: 1
---

このセクションでは、W&B を導入するさまざまな方法について説明します。

## W&B マルチテナントクラウド
[W&B Multi-tenant Cloud]({{< relref "saas_cloud.md" >}}) は、W&B により完全に管理されており、アップグレード、メンテナンス、プラットフォームのセキュリティやキャパシティプランニングまで対応しています。マルチテナントクラウドは、W&B の Google Cloud Platform (GCP) アカウント上、[GCP の北米リージョン](https://cloud.google.com/compute/docs/regions-zones)にデプロイされます。[お客様自身のバケット（BYOB）]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) 機能により、W&B Artifacts やその他の機密データを、お客様自身のクラウド、またはオンプレミスのインフラストラクチャーに保存することも可能です。

詳細は [W&B Multi-tenant Cloud]({{< relref "saas_cloud.md" >}}) をご覧いただくか、[無料で始める](https://app.wandb.ai/login?signup=true)をご利用ください。

## W&B Dedicated Cloud
[W&B Dedicated Cloud]({{< relref "dedicated_cloud/" >}}) は、エンタープライズ向けに設計されたシングルテナント型の完全マネージドプラットフォームです。W&B Dedicated Cloud は、W&B の AWS、GCP、または Azure アカウント上にデプロイされます。Dedicated Cloud は、マルチテナントクラウドより柔軟性が高く、W&B Self-Managed よりは運用が簡単です。アップグレード、メンテナンス、プラットフォームのセキュリティ、キャパシティプランニングは W&B が管理します。各 Dedicated Cloud インスタンスは、他の Dedicated Cloud インスタンスと分離されたネットワーク・コンピューティング・ストレージ環境で提供されます。

お客様の W&B 固有のメタデータやデータは、分離されたクラウドストレージに保存され、分離されたクラウドコンピュートサービスで処理されます。[お客様自身のバケット（BYOB）]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) 機能を利用すれば、アーティファクトやその他の関連機密データを、お客様自身のクラウドまたはオンプレミスのインフラストラクチャ―に保存することも可能です。

W&B Dedicated Cloud には、重要なセキュリティやエンタープライズ向け機能サポートを含む [エンタープライズライセンス]({{< relref "self-managed/server-upgrade-process.md" >}}) が付属します。

さらに 高度なセキュリティやコンプライアンス要件を持つ組織には、HIPAA 準拠、シングルサインオン、またはカスタマー管理暗号鍵（CMEK）など **エンタープライズ** サポートのもとでご利用になれます。[詳細はこちらからお問い合わせください](https://wandb.ai/site/contact)。

詳細は [W&B Dedicated Cloud]({{< relref "dedicated_cloud/" >}}) をご覧いただくか、[無料で始める](https://app.wandb.ai/login?signup=true)をご利用ください。

## W&B Self-Managed
[W&B Self-Managed]({{< relref "self-managed/" >}}) は、ご自身で完全に管理するタイプで、お客様のオンプレミスまたはご自身で管理するクラウド インフラ上で運用可能です。IT、DevOps、MLOps チームの皆さまが担当する内容は以下の通りです。
- デプロイメントの準備
- 国際標準や[セキュリティ技術実装ガイド（STIG）](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide)に沿ったインフラストラクチャーのセキュリティ確保（該当する場合）
- アップグレード・パッチ管理
- W&B サーバーの継続的な運用・保守

ご希望に応じて、W&B Self-Managed のエンタープライズライセンスの取得が可能です。エンタープライズライセンスには、重要なセキュリティやエンタープライズ向け機能へのサポートが含まれます。

詳細については [W&B Self-Managed]({{< relref "self-managed/" >}}) や、[リファレンスアーキテクチャー]({{< relref "self-managed/ref-arch.md" >}}) ガイドラインをご覧ください。