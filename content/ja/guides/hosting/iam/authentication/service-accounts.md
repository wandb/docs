---
title: Use service accounts to automate workflows
description: 自動または非対話型のワークフローを管理する際に、組織とチームのスコープを持つサービスアカウントを使用する
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-service-accounts
---

サービスアカウントは、非ヒューマンまたは機械ユーザーを表し、チーム内や複数のチームにわたるプロジェクトで一般的なタスクを自動的に実行できます。

- 組織の管理者は、組織の範囲でサービスアカウントを作成できます。
- チームの管理者は、そのチームの範囲でサービスアカウントを作成できます。

サービスアカウントの APIキーは、呼び出し元がサービスアカウントの範囲内のプロジェクトに対して読み書きを行うことを可能にします。

サービスアカウントは、複数のユーザーまたはチームによるワークフローの集中管理を可能にし、W&B Models の実験管理を自動化したり、W&B Weave のトレースをログに記録したりできます。サービスアカウントによって管理されるワークフローにユーザーの身元を関連付けるオプションがあり、[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` のいずれかを使用します。

{{% alert %}}
サービスアカウントは、 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、エンタープライズライセンスを持つ [Self-managed instances]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})、および [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) のエンタープライズアカウントで利用可能です。
{{% /alert %}}

## 組織範囲のサービスアカウント

組織にスコープされたサービスアカウントは、チームに関係なく、組織内のすべてのプロジェクトへの読み書き権限を持っています。ただし、[制限付きプロジェクト]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) は例外です。組織範囲のサービスアカウントが制限付きプロジェクトにアクセスする前に、そのプロジェクトの管理者が明示的にサービスアカウントをプロジェクトに追加する必要があります。

組織管理者は、組織またはアカウントのダッシュボードの **Service Accounts** タブから、組織範囲のサービスアカウントの APIキーを取得できます。

新しい組織範囲のサービスアカウントを作成するには:

* 組織のダッシュボードの **Service Accounts** タブで、**New service account** ボタンをクリックします。
* **Name** を入力します。
* デフォルトのチームをサービスアカウントに選択します。
* **Create** をクリックします。
* 新しく作成されたサービスアカウントの横にある **Copy API key** をクリックします。
* コピーした APIキーをシークレットマネージャーや他の安全でアクセス可能な場所に保存します。

{{% alert %}}
組織範囲のサービスアカウントは、組織内のすべてのチームによって所有される非制限プロジェクトにアクセスできるにもかかわらず、デフォルトのチームを必要とします。これは、モデルトレーニングまたは生成的AIアプリの環境で `WANDB_ENTITY` 変数が設定されていない場合に、ワークロードが失敗するのを防ぐのに役立ちます。別のチームにあるプロジェクトに組織範囲のサービスアカウントを使用するには、`WANDB_ENTITY` 環境変数をそのチームに設定する必要があります。
{{% /alert %}}

## チーム範囲のサービスアカウント

チームにスコープされたサービスアカウントは、そのチーム内のすべてのプロジェクトで、[制限付きプロジェクト]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}})を除いて読み書きができます。チームにスコープされたサービスアカウントが制限付きプロジェクトにアクセスする前に、そのプロジェクトの管理者が明示的にサービスアカウントをプロジェクトに追加する必要があります。

チーム管理者として、`<WANDB_HOST_URL>/<your-team-name>/service-accounts` にアクセスして、チーム内のチームスコープサービスアカウントの APIキーを取得できます。また、チームの **Team settings** で **Service Accounts** タブを参照することもできます。

チームのための新しいチームスコープサービスアカウントを作成するには:

* チームの **Service Accounts** タブで、**New service account** ボタンをクリックします。
* **Name** を入力します。
* 認証方法として **Generate API key (Built-in)** を選択します。
* **Create** をクリックします。
* 新しく作成されたサービスアカウントの横にある **Copy API key** をクリックします。
* コピーした APIキーをシークレットマネージャーや他の安全でアクセス可能な場所に保存します。

チームにスコープされたサービスアカウントを使用するモデルトレーニングまたは生成的AIアプリの環境でチームを設定しない場合、モデルの run や weave トレースは、サービスアカウントの親チーム内の指定されたプロジェクトにログ記録されます。このようなシナリオでは、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 変数を使用したユーザーの帰属は、参照されるユーザーがサービスアカウントの親チームの一部でない限り、_機能しません_。

{{% alert color="warning" %}}
チームにスコープされたサービスアカウントは、親チームと異なるチームの [team or restricted-scoped project]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) に run をログに記録することはできませんが、開放された可視性プロジェクトに対しては、他のチーム内で run をログに記録できます。
{{% /alert %}}

### 外部サービスアカウント

**Built-in** サービスアカウントに加えて、W&B は、W&B SDK と CLI を使用して [Identity federation]({{< relref path="./identity_federation.md#external-service-accounts" lang="ja" >}}) を使用する、JSON Webトークン (JWTs) を発行できるアイデンティティプロバイダー (IdPs) によるチームスコープの **External service accounts** もサポートしています。