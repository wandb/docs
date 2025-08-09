---
title: デプロイメントのオプション
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-_index
    parent: w-b-platform
weight: 1
---

このセクションでは、W&B をデプロイするさまざまな方法について説明します。

## W&B Multi-tenant Cloud
[W&B Multi-tenant Cloud]({{< relref path="saas_cloud.md" lang="ja" >}}) は、アップグレード、メンテナンス、プラットフォームのセキュリティ、キャパシティプランニングまで W&B が完全に管理するサービスです。Multi-tenant Cloud は W&B の Google Cloud Platform (GCP) アカウント上で [GCP の北米リージョン](https://cloud.google.com/compute/docs/regions-zones)にデプロイされます。[Bring your own bucket (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を利用すると、W&B Artifacts やその他の機密データを自社のクラウドまたはオンプレミスのインフラストラクチャーに保存することも可能です。

詳しくは [W&B Multi-tenant Cloud]({{< relref path="saas_cloud.md" lang="ja" >}}) または [無料で始める](https://app.wandb.ai/login?signup=true) をご覧ください。

## W&B Dedicated Cloud
[W&B Dedicated Cloud]({{< relref path="dedicated_cloud/" lang="ja" >}}) は、エンタープライズ組織向けに設計されたシングルテナントの完全管理型プラットフォームです。W&B Dedicated Cloud は、W&B の AWS、GCP、Azure のいずれかのアカウントにデプロイされます。Dedicated Cloud は Multi-tenant Cloud よりも柔軟性が高く、W&B Self-Managed よりもシンプルに利用できます。アップグレードやメンテナンス、プラットフォームのセキュリティ、キャパシティプランニングも W&B が管理します。各 Dedicated Cloud インスタンスは、他の W&B Dedicated Cloud インスタンスから独立したネットワーク・コンピュート・ストレージを持っています。

W&B 固有のメタデータやデータは、分離されたクラウドストレージに保存され、分離されたクラウドコンピュートサービスで処理されます。[Bring your own bucket (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を利用すれば、Artifacts やその他関連する機密データを自社のクラウドやオンプレミスのインフラストラクチャーに保存することも可能です。

W&B Dedicated Cloud には、重要なセキュリティやエンタープライズ向け機能のサポートが含まれる [エンタープライズライセンス]({{< relref path="self-managed/server-upgrade-process.md" lang="ja" >}}) が付属します。

高度なセキュリティやコンプライアンス要件をお持ちの組織向けに、HIPAA 準拠、シングルサインオン、Customer Managed Encryption Keys (CMEK) などの機能も **Enterprise** サポートとして利用可能です。[詳細はこちらからお問い合わせください](https://wandb.ai/site/contact)。

詳しくは [W&B Dedicated Cloud]({{< relref path="dedicated_cloud/" lang="ja" >}}) または [無料で始める](https://app.wandb.ai/login?signup=true) をご覧ください。

## W&B Self-Managed
[W&B Self-Managed]({{< relref path="self-managed/" lang="ja" >}}) は、完全にお客様自身で管理し、オンプレミスやお客様が管理するクラウドインフラ上に構築できます。貴社の IT/DevOps/MLOps チームが以下を担当します：
- デプロイメントの準備
- 貴社のポリシーや [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) 適用に基づいたインフラストラクチャーのセキュリティ
- アップグレードやパッチの管理
- Self-Managed W&B Server インスタンスの継続的な保守

W&B Self-Managed 用のエンタープライズライセンスをオプションで取得することも可能です。エンタープライズライセンスには、重要なセキュリティやエンタープライズ向け機能のサポートが含まれます。

詳しくは [W&B Self-Managed]({{< relref path="self-managed/" lang="ja" >}}) または [reference architecture]({{< relref path="self-managed/ref-arch.md" lang="ja" >}}) のガイドラインもご参照ください。