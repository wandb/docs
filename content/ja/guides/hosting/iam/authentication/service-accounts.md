---
title: Use service accounts to automate workflows
description: 組織および Team スコープのサービスアカウントを使用して、自動化された、あるいは非インタラクティブな ワークフロー を管理します。
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-service-accounts
---

サービスアカウントは、チーム内または複数のチームにわたって、プロジェクト全体の一般的なタスクを自動的に実行できる、人間以外の ユーザー またはマシン ユーザー を表します。

- 組織の管理者は、組織のスコープでサービスアカウントを作成できます。
- チーム管理者は、そのチームのスコープでサービスアカウントを作成できます。

サービスアカウントの APIキー を使用すると、呼び出し元はサービスアカウントのスコープ内の project に対して読み取りまたは書き込みを行うことができます。

サービスアカウントを使用すると、複数の ユーザー または Teams による ワークフロー の集中管理、W&B Models の 実験管理 の自動化、または W&B Weave の トレース の ログ 記録が可能になります。[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` のいずれかを使用すると、サービスアカウントによって管理される ワークフロー に人間の ユーザー の ID を関連付けるオプションがあります。

{{% alert %}}
サービスアカウントは、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、エンタープライズライセンス付きの[自己管理インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})、および[SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})のエンタープライズアカウントで利用できます。
{{% /alert %}}

## 組織スコープのサービスアカウント

組織をスコープとするサービスアカウントは、[制限付き project ]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}})を除き、Teams に関係なく、組織内のすべての project で読み取りおよび書き込みを行う権限を持ちます。組織をスコープとするサービスアカウントが制限付き project に アクセス するには、その project の管理者がサービスアカウントを project に明示的に追加する必要があります。

組織管理者は、組織またはアカウント ダッシュボード の **Service Accounts** タブから、組織をスコープとするサービスアカウントの APIキー を取得できます。

新しい組織スコープのサービスアカウントを作成するには:

* 組織 ダッシュボード の **Service Accounts** タブにある **New service account** ボタンをクリックします。
* **Name** を入力します。
* サービスアカウントのデフォルト Teams を選択します。
* **Create** をクリックします。
* 新しく作成したサービスアカウントの横にある **Copy API key** をクリックします。
* コピーした APIキー を、シークレットマネージャーまたはその他の安全で アクセス 可能な場所に保存します。

{{% alert %}}
組織をスコープとするサービスアカウントには、組織内のすべての Teams が所有する制限されていない project への アクセス 権がある場合でも、デフォルト Teams が必要です。これは、モデル トレーニング または生成 AI アプリの 環境 で `WANDB_ENTITY` 変数が設定されていない場合に ワークロード が失敗するのを防ぐのに役立ちます。別の Teams の project に組織をスコープとするサービスアカウントを使用するには、`WANDB_ENTITY` 環境 変数をその Teams に設定する必要があります。
{{% /alert %}}

## チームスコープのサービスアカウント

Teams をスコープとするサービスアカウントは、その Teams 内の[制限付き project ]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}})を除き、その Teams 内のすべての project で読み取りおよび書き込みを行うことができます。Teams をスコープとするサービスアカウントが制限付き project に アクセス するには、その project の管理者がサービスアカウントを project に明示的に追加する必要があります。

Teams 管理者として、`<WANDB_HOST_URL>/<your-team-name>/service-accounts` にある Teams 内の Teams スコープのサービスアカウントの APIキー を取得できます。または、Teams の **Team settings** に移動し、**Service Accounts** タブを参照することもできます。

Teams の新しい Teams スコープのサービスアカウントを作成するには:

* Teams の **Service Accounts** タブにある **New service account** ボタンをクリックします。
* **Name** を入力します。
* 認証 method として **Generate API key (Built-in)** を選択します。
* **Create** をクリックします。
* 新しく作成したサービスアカウントの横にある **Copy API key** をクリックします。
* コピーした APIキー を、シークレットマネージャーまたはその他の安全で アクセス 可能な場所に保存します。

Teams スコープのサービスアカウントを使用するモデル トレーニング または生成 AI アプリの 環境 で Teams を構成しない場合、モデル run または weave トレース は、サービスアカウントの親 Teams 内の名前付き project に ログ 記録されます。このようなシナリオでは、参照 ユーザー がサービスアカウントの親 Teams の一部でない限り、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 変数を使用した ユーザー 属性は _機能しません_。

{{% alert color="warning" %}}
Teams スコープのサービスアカウントは、親 Teams とは異なる Teams の [Teams または制限付きスコープの project ]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}})に run を ログ 記録できませんが、別の Teams 内のオープンな可視性 project に run を ログ 記録できます。
{{% /alert %}}

### 外部サービスアカウント

**Built-in** サービスアカウントに加えて、W&B は、JSON Web Tokens (JWT) を発行できる ID プロバイダー (IdP) との[ID フェデレーション]({{< relref path="./identity_federation.md#external-service-accounts" lang="ja" >}})を使用して、W&B SDK および CLI を使用した Teams スコープの **External service accounts** もサポートしています。
