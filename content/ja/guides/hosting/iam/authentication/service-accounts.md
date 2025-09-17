---
title: サービス アカウントを使用してワークフローを自動化する
description: 組織およびチーム スコープのサービス アカウントを使用して、自動化または非対話型のワークフローを管理する
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-service-accounts
---

サービスアカウントは、人ではない機械ユーザーを表し、team 内の projects 間や複数の teams にわたってよくある作業を自動で実行できます。サービスアカウントは CI/CD パイプライン、自動トレーニング ジョブ、その他のマシン間ワークフローに最適です。

## 主なメリット

{{< readfile file="/content/en/_includes/service-account-benefits.md" >}}

## 概要

サービスアカウントは、個人のユーザー資格情報やハードコーディングした資格情報を使わずに、W&B のワークフローを自動化する安全な方法を提供します。作成できるスコープは 2 種類です。

- **Organization-scoped**: org 管理者が作成し、すべての teams にまたがってアクセスできます。
- **Team-scoped**: team 管理者が作成し、特定の team にのみアクセスが制限されます。
	
サービスアカウントの APIキー により、呼び出し元はサービスアカウントのスコープ内の projects に対して読み書きできます。これにより、W&B Models における 実験管理 の自動ワークフローや、W&B Weave におけるトレースのログを一元的に管理できます。

サービスアカウントは次の用途で特に有用です。
- **CI/CD パイプライン**: GitHub Actions、GitLab CI、Jenkins からモデルトレーニング runs を自動でログ
- **スケジュール済みジョブ**: 毎晩のモデル再トレーニング、定期的な評価 runs、データ検証ワークフロー
- **プロダクション監視**: 本番システムから推論メトリクスや model のパフォーマンスをログ
- **Jupyter notebooks**: JupyterHub や Google Colab の共有ノートブック
- **Kubernetes ジョブ**: K8s クラスターで動く自動ワークフロー
- **Airflow/Prefect/Dagster**: ML パイプラインのオーケストレーション ツール

{{% alert %}}
サービスアカウントは、エンタープライズ ライセンスをお持ちの場合の [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、[セルフマネージド インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})、および [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) のエンタープライズ アカウントでご利用いただけます。
{{% /alert %}}

## Organization-scoped サービスアカウント

organization にスコープされたサービスアカウントは、team に関わらず organization 内のすべての projects に対して読み書きの権限があります。ただし [restricted projects]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) は例外です。organization-scoped のサービスアカウントが restricted project にアクセスするには、その project の管理者がサービスアカウントを明示的にその project に追加する必要があります。

organization 管理者は、organization またはアカウントのダッシュボードの **Service Accounts** タブから、organization-scoped サービスアカウントの APIキー を取得できます。

新しい organization-scoped サービスアカウントを作成するには:

* organization のダッシュボードの **Service Accounts** タブで **New service account** をクリックします。
* **Name** を入力します。
* そのサービスアカウントのデフォルト team を選択します。
* **Create** をクリックします。
* 作成したサービスアカウントの右側にある **Copy API key** をクリックします。
* コピーした APIキー をシークレットマネージャーなど、安全かつアクセス可能な場所に保管します。

{{% alert %}}
organization-scoped のサービスアカウントは、organization 内のすべての teams が所有する非制限の projects にアクセスできる場合でも、デフォルトの team が必要です。これは、モデルトレーニングや生成 AI アプリの 環境 で `WANDB_ENTITY` 変数が設定されていないときにワークロードが失敗するのを防ぐためです。別の team の project で organization-scoped サービスアカウントを使うには、その team を指すように `WANDB_ENTITY` 環境変数を設定してください。
{{% /alert %}}

## Team-scoped サービスアカウント

team にスコープされたサービスアカウントは、その team 内のすべての projects に対して読み書きできますが、その team の [restricted projects]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) は除きます。team-scoped のサービスアカウントが restricted project にアクセスするには、その project の管理者がサービスアカウントを明示的にその project に追加する必要があります。

team 管理者は、自分の team の `<WANDB_HOST_URL>/<your-team-name>/service-accounts` で team-scoped サービスアカウントの APIキー を取得できます。あるいは **Team settings** に移動して **Service Accounts** タブを参照することもできます。

自分の team に新しい team-scoped サービスアカウントを作成するには:

* team の **Service Accounts** タブで **New service account** をクリックします。
* **Name** を入力します。
* 認証 method として **Generate API key (Built-in)** を選択します。
* **Create** をクリックします。
* 作成したサービスアカウントの右側にある **Copy API key** をクリックします。
* コピーした APIキー をシークレットマネージャーなど、安全かつアクセス可能な場所に保管します。

team-scoped サービスアカウントを使うモデルトレーニングや生成 AI アプリの 環境 に team を設定しない場合、model runs や Weave traces は、そのサービスアカウントの親 team にある指定の project にログされます。このような状況では、参照しているユーザーがそのサービスアカウントの親 team の一員でない限り、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 変数を使ったユーザー帰属は _機能しません_。

{{% alert color="warning" %}}
team-scoped のサービスアカウントは、親 team とは異なる team にある [team または restricted スコープの project]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) に runs をログできませんが、別の team 内の公開範囲が open の project には runs をログできます。
{{% /alert %}}

### 外部サービスアカウント

組み込みのサービスアカウントに加えて、W&B は W&B SDK と CLI による [アイデンティティ フェデレーション]({{< relref path="./identity_federation.md#external-service-accounts" lang="ja" >}}) を用いて、JSON Web Token (JWT) を発行できる IdP（アイデンティティ プロバイダ）との連携による team-scoped の外部サービスアカウントもサポートします。

## ベストプラクティス

組織でサービスアカウントを安全かつ効率的に使うため、次の推奨事項に従ってください。

- **シークレットマネージャーを使用する**: サービスアカウントの APIキー は、プレーンテキストの設定ファイルではなく、AWS Secrets Manager、HashiCorp Vault、Azure Key Vault などの安全なシークレット管理システムに保存します。

- **最小権限の原則**: 必要な projects への アクセス のみに限定するため、可能であれば organization-scoped ではなく team-scoped のサービスアカウントを作成します。

- **ユースケースごとに専用のサービスアカウント**: 監査性を高め、きめ細かな アクセス 制御を可能にするため、（例: CI/CD 用と定期的な再トレーニング用で）自動化ワークフローごとに別のサービスアカウントを作成します。

- **定期的な監査**: 稼働中のサービスアカウントを定期的に見直し、不要になったものは削除します。監査ログを確認して、サービスアカウントの活動をモニタリングします。

- **APIキー の安全な取り扱い**: 
  - APIキー をバージョン管理にコミットしない
  - アプリケーションにキーを渡すときは環境変数を使用する
  - うっかり漏洩した場合はキーをローテーションする

- **命名規則**: サービスアカウントの目的が分かる記述的な名前を使用します:
  - Good: `ci-model-training`, `nightly-eval-pipeline`, `prod-inference-monitor`
  - Avoid: `service-account-1`, `test-sa`, `temp`

- **ユーザー帰属**: 複数の チームメンバー が同じ自動化ワークフローを使う場合、誰が各 run をトリガーしたかを追跡できるように `WANDB_USERNAME` または `WANDB_USER_EMAIL` を設定します:
  ```bash
  export WANDB_API_KEY="<service_account_key>"
  export WANDB_USERNAME="john.doe@company.com"
  ```

- **環境の設定**: team-scoped サービスアカウントでは、runs が正しい team にログされるよう常に `WANDB_ENTITY` を設定します:
  ```bash
  export WANDB_ENTITY="ml-team"
  export WANDB_PROJECT="production-models"
  ```

- **エラー処理**: 失敗した認証に対して適切なエラー処理とアラートを実装し、サービスアカウントの資格情報に関する問題を素早く特定できるようにします。

- **ドキュメント**: 次の情報を維持・管理します:
  - どのサービスアカウントが存在し、その目的は何か
  - 各サービスアカウントをどのシステム/ワークフローが使用しているか
  - 各アカウントを担当する team の連絡先情報

## トラブルシューティング

よくある問題と解決策:

- **"Unauthorized" エラー**: APIキー が正しく設定され、サービスアカウントが対象の project に アクセス できることを確認する
- **runs が表示されない**: `WANDB_ENTITY` が正しい team 名に設定されているか確認する
- **ユーザー帰属が機能しない**: `WANDB_USERNAME` で指定したユーザーがその team のメンバーであることを確認する
- **restricted projects への アクセス 拒否**: サービスアカウントを restricted project の アクセス リストに明示的に追加する