---
title: Use service accounts to automate workflows
description: 組織とチームのスコープを持つサービスアカウントを使用して、自動化された、または非インタラクティブな ワークフロー を管理します。
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-service-accounts
---

サービスアカウントは、チーム内またはチーム間で、プロジェクトを横断して共通タスクを自動的に実行できる、人間ではないまたは機械の ユーザー を表します。

- 組織の 管理者 は、組織の スコープ で サービスアカウント を作成できます。
- チーム の 管理者 は、その チーム の スコープ で サービスアカウント を作成できます。

サービスアカウント の APIキー を使用すると、呼び出し元は サービスアカウント の スコープ 内の プロジェクト から読み取りまたは書き込みができます。

サービスアカウント を使用すると、複数の ユーザー または チーム による ワークフロー の集中管理、W&B Models の 実験管理 の自動化、または W&B Weave の トレース の ログ記録 が可能になります。 [ 環境変数 ]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` のいずれかを使用することにより、人間の ユーザー の ID を サービスアカウント によって管理される ワークフロー に関連付けるオプションがあります。

{{% alert %}}
サービスアカウント は、エンタープライズライセンスによる [ Dedicated Cloud ]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、[ Self-managed instances ]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) 、および [ SaaS Cloud ]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) のエンタープライズアカウントで利用できます。
{{% /alert %}}

## 組織スコープ の サービスアカウント

組織に スコープ された サービスアカウント は、[ 制限付き プロジェクト ]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) を除き、 チーム に関係なく、組織内のすべての プロジェクト で読み取りおよび書き込みの権限を持ちます。組織スコープ の サービスアカウント が 制限付き プロジェクト に アクセス するには、その プロジェクト の 管理者 が サービスアカウント を プロジェクト に明示的に追加する必要があります。

組織の 管理者 は、組織またはアカウント ダッシュボード の [ **Service Accounts** ] タブから、組織スコープ の サービスアカウント の APIキー を取得できます。

新しい組織スコープ の サービスアカウント を作成するには:

* 組織 ダッシュボード の [ **Service Accounts** ] タブにある [ **New service account** ] ボタンをクリックします。
* [ **Name** ] を入力します。
* サービスアカウント のデフォルト チーム を選択します。
* [ **Create** ] をクリックします。
* 新しく作成された サービスアカウント の横にある [ **Copy API key** ] をクリックします。
* コピーした APIキー を、シークレットマネージャーまたはその他の安全で アクセス 可能な場所に保存します。

{{% alert %}}
組織スコープ の サービスアカウント には、組織内のすべての チーム が所有する制限のない プロジェクト に アクセス できる場合でも、デフォルト チーム が必要です。これにより、モデルトレーニング または 生成AI アプリ の 環境 で `WANDB_ENTITY` 変数 が 設定 されていない場合に、 ワークロード が失敗するのを防ぐことができます。別の チーム の プロジェクト で組織スコープ の サービスアカウント を使用するには、`WANDB_ENTITY` 環境変数 をその チーム に設定する必要があります。
{{% /alert %}}

## チームスコープ の サービスアカウント

チームスコープ の サービスアカウント は、その チーム 内のすべての プロジェクト で読み取りおよび書き込みができます。ただし、その チーム 内の [ 制限付き プロジェクト ]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) は除きます。チームスコープ の サービスアカウント が 制限付き プロジェクト に アクセス するには、その プロジェクト の 管理者 が サービスアカウント を プロジェクト に明示的に追加する必要があります。

チーム の 管理者 として、チームスコープ の サービスアカウント の APIキー を `<WANDB_HOST_URL>/<your-team-name>/service-accounts` で取得できます。または、チーム の [ **Team settings** ] に移動し、[ **Service Accounts** ] タブを参照することもできます。

チーム の新しい チーム スコープ の サービスアカウント を作成するには:

* チーム の [ **Service Accounts** ] タブにある [ **New service account** ] ボタンをクリックします。
* [ **Name** ] を入力します。
* 認証 方法として [ **Generate API key (Built-in)** ] を選択します。
* [ **Create** ] をクリックします。
* 新しく作成された サービスアカウント の横にある [ **Copy API key** ] をクリックします。
* コピーした APIキー を、シークレットマネージャーまたはその他の安全で アクセス 可能な場所に保存します。

チームスコープ の サービスアカウント を使用する モデルトレーニング または 生成AI アプリ 環境 で チーム を構成しない場合、モデル の run または Weave トレース は、 サービスアカウント の親 チーム 内の名前付き プロジェクト に ログ 記録 されます。このようなシナリオでは、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 変数 を使用した ユーザー 属性は、参照される ユーザー が サービスアカウント の親 チーム の一部である場合を除き、_機能しません_。

{{% alert color="warning" %}}
チームスコープ の サービスアカウント は、親 チーム とは異なる チーム の [ チーム または 制限 付き スコープ の プロジェクト ]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) に run を ログ 記録 できませんが、別の チーム 内の 公開 可視性 プロジェクト に run を ログ 記録 できます。
{{% /alert %}}

### 外部 サービスアカウント

**Built-in** サービスアカウント に加えて、W&B は、JSON Web Tokens (JWT) を発行できる ID プロバイダー (IdP) との [ Identity federation ]({{< relref path="./identity_federation.md#external-service-accounts" lang="ja" >}}) を使用して、W&B SDK および CLI を使用した チーム スコープ の **External service accounts** もサポートしています。
