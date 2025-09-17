---
title: W&B のマルチテナント クラウドを利用する
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-saas_cloud
    parent: deployment-options
weight: 1
---

W&B Multi-tenant Cloud は、W&B の Google Cloud Platform (GCP) アカウント上の [GCP の北米リージョン](https://cloud.google.com/compute/docs/regions-zones) にデプロイされた、フルマネージドな プラットフォーム です。GCP のオートスケーリングを活用し、トラフィックの増減に応じて プラットフォーム が適切にスケールします。

{{< img src="/images/hosting/saas_cloud_arch.png" alt="Multi-tenant Cloud アーキテクチャー図" >}}

W&B Multi-tenant Cloud は組織のニーズに合わせてスケールし、プロジェクト ごとに最大 250,000 個のメトリクスをログでき、メトリクスあたり最大 100 万個の データ ポイントをサポートします。より大規模な デプロイメント については、[サポート](mailto:support@wandb.com) にお問い合わせください。

## データ セキュリティ

Free または Pro プランの ユーザー の場合、すべての データ は共有 クラウド ストレージにのみ保存され、共有 クラウド コンピューティング サービスで処理されます。お客様の料金プランによっては、ストレージ制限の対象となる場合があります。

Enterprise プランの ユーザー は、[セキュア ストレージ コネクターを使用して独自の バケット (BYOB) を持ち込む]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) ことができ、[チーム レベル]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}}) で Models、Datasets などのファイルを保存できます。複数の Teams で単一の バケット を 設定 するか、異なる W&B Teams で個別の バケット を使用できます。チームのために BYOB を 設定 しない場合、チームの データ は共有 クラウド ストレージに保存されます。

お客様の デプロイメント が組織のポリシーに準拠していることを確認する責任があります。また、該当する場合は [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) にも準拠していることを確認してください。

## ID および アクセス 管理 (IAM)
Enterprise プランをご利用の場合、強化された ID および アクセス 管理機能により、W&B デプロイメント の安全な認証と効果的な認可が可能になります。

*   OIDC または SAML を使用した SSO 認証。組織の SSO を 設定 したい場合は、W&B Teams または サポート にお問い合わせください。
*   組織全体およびチーム内で、[適切な ユーザー ロールを 設定]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}}) します。
*   W&B Projects の範囲を定義して、誰が Projects を表示、編集、W&B Runs を送信できるかを制限します。その際は [制限付き Projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を使用します。

## 監視
組織の管理者は、アカウント ビューの `Billing` タブからアカウントの使用状況と請求を管理できます。Multi-tenant Cloud で共有 クラウド ストレージを使用している場合、管理者は組織内の異なる Teams 間でストレージ使用量を最適化できます。

## メンテナンス
W&B Multi-tenant Cloud は、マルチテナントのフルマネージドな プラットフォーム です。W&B によって管理されているため、W&B プラットフォーム のプロビジョニングとメンテナンスにかかるオーバーヘッドやコストは発生しません。

## コンプライアンス
Multi-tenant Cloud のセキュリティ管理は、定期的に内部および外部監査を受けています。SOC 2 レポートおよびその他のセキュリティとコンプライアンスに関するドキュメントを要求するには、[W&B Security Portal](https://security.wandb.ai/) を参照してください。

## 次のステップ
ほとんどの機能を無料で開始するには、[Multi-tenant Cloud に直接 アクセス](https://wandb.ai) してください。強化された データ セキュリティ と IAM 機能を試すには、[Enterprise トライアル をリクエスト](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial) してください。