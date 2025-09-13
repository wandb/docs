---
title: デプロイ オプション
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-_index
    parent: w-b-platform
weight: 1
---

W&B のデプロイ方法について説明します。

## W&B Multi-tenant Cloud
[W&B Multi-tenant Cloud]({{< relref path="saas_cloud.md" lang="ja" >}}) は、アップグレード、メンテナンス、プラットフォーム セキュリティ、キャパシティ プランニングを含め、W&B が完全に管理します。Multi-tenant Cloud は、W&B の Google Cloud Platform (GCP) アカウントの [GCP の北米リージョン](https://cloud.google.com/compute/docs/regions-zones) にデプロイされます。[Bring your own bucket (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) は、オプションで W&B Artifacts およびその他の機密性の高い関連データを、ご自身のクラウドまたはオンプレミス インフラストラクチャーに保存できます。

[W&B Multi-tenant Cloud]({{< relref path="saas_cloud.md" lang="ja" >}}) を参照するか、[無料で開始](https://app.wandb.ai/login?signup=true) してください。

## W&B Dedicated Cloud
[W&B Dedicated Cloud]({{< relref path="dedicated_cloud/" lang="ja" >}}) は、エンタープライズ組織を念頭に設計された、シングルテナントのフルマネージド プラットフォームです。W&B Dedicated Cloud は、W&B の AWS、GCP、または Azure アカウントにデプロイされます。Dedicated Cloud は、Multi-tenant Cloud よりも高い柔軟性を提供しますが、W&B Self-Managed よりも複雑さは少なくなります。アップグレード、メンテナンス、プラットフォーム セキュリティ、キャパシティ プランニングは W&B が管理します。各 Dedicated Cloud インスタンスは、他の W&B Dedicated Cloud インスタンスから分離された独自のネットワーク、コンピューティング、およびストレージを持っています。

W&B 固有のメタデータとデータは、分離されたクラウド ストレージに保存され、分離されたクラウド コンピューティング サービスを使用して処理されます。[Bring your own bucket (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) は、オプションで Artifacts およびその他の機密性の高い関連データを、ご自身のクラウドまたはオンプレミス インフラストラクチャーに保存できます。

W&B Dedicated Cloud には、重要なセキュリティ機能やその他のエンタープライズ向けの機能に対応する [エンタープライズ ライセンス]({{< relref path="self-managed/server-upgrade-process.md" lang="ja" >}}) が含まれています。

高度なセキュリティまたはコンプライアンス要件を持つ組織向けには、HIPAA コンプライアンス、シングル サインオン、顧客管理型暗号化キー (CMEK) などの機能が **Enterprise** サポートで利用可能です。[詳細情報をリクエスト](https://wandb.ai/site/contact) してください。

[W&B Dedicated Cloud]({{< relref path="dedicated_cloud/" lang="ja" >}}) を参照するか、[無料で開始](https://app.wandb.ai/login?signup=true) してください。

## W&B Self-Managed
[W&B Self-Managed]({{< relref path="self-managed/" lang="ja" >}}) は、お客様のオンプレミス環境、またはお客様が管理するクラウド インフラストラクチャーのいずれかで、お客様が完全に管理します。お客様の IT/DevOps/MLOps チームは、以下を担当します。
- デプロイメントのプロビジョニング。
- 該当する場合、組織のポリシーおよび [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) に従って、インフラストラクチャーを保護する。
- アップグレードの管理とパッチの適用。
- 自己管理型の W&B Server インスタンスを継続的に維持する。

W&B Self-Managed 用のエンタープライズ ライセンスをオプションで取得できます。エンタープライズ ライセンスには、重要なセキュリティ機能やその他のエンタープライズ向けの機能のサポートが含まれています。

[W&B Self-Managed]({{< relref path="self-managed/" lang="ja" >}}) を参照するか、[リファレンス アーキテクチャー]({{< relref path="self-managed/ref-arch.md" lang="ja" >}}) ガイドラインを確認してください。