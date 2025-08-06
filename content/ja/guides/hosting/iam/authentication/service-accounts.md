---
title: サービスアカウントを使用してワークフローを自動化する
description: 組織やチーム単位のサービスアカウントを使って、自動化または非対話型のワークフローを管理する
displayed_sidebar: default
---

サービスアカウントは、チーム内やチーム間でプロジェクトを自動的に操作できる非人間または機械ユーザーを表します。

- 組織管理者は、組織スコープでサービスアカウントを作成できます。
- チーム管理者は、そのチームのスコープでサービスアカウントを作成できます。
	
サービスアカウントの APIキー を使用すると、そのサービスアカウントのスコープ内のプロジェクトに対して読み書きできます。

サービスアカウントを利用することで、複数ユーザーやチームによるワークフローを一元的に管理し、 W&B Models の実験管理や W&B Weave のトレースログを自動化できます。さらに、 [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` を利用して、サービスアカウントが管理するワークフローに人間ユーザーの識別情報を関連付けることも可能です。

{{% alert %}}
サービスアカウントは [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}})、エンタープライズライセンスを持つ [Self-managed instances]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}})、および [SaaS Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) のエンタープライズアカウントでご利用いただけます。
{{% /alert %}}

## 組織スコープのサービスアカウント

組織スコープのサービスアカウントは、チームに関係なく組織内のすべてのプロジェクトで読み書きできます（[制限付きプロジェクト]({{< relref "../access-management/restricted-projects.md#visibility-scopes" >}}) を除く）。組織スコープのサービスアカウントが制限付きプロジェクトにアクセスするには、そのプロジェクトの管理者が明示的にサービスアカウントをプロジェクトに追加する必要があります。

組織管理者は、組織またはアカウントの ダッシュボード の **Service Accounts** タブから、組織スコープのサービスアカウントの APIキー を取得できます。

新しい組織スコープのサービスアカウントを作成するには：

* 組織ダッシュボードの **Service Accounts** タブで **New service account** ボタンをクリックします。
* **Name** を入力します。
* サービスアカウントのデフォルトチームを選択します。
* **Create** をクリックします。
* 新しく作成されたサービスアカウントの横で **Copy API key** をクリックします。
* コピーした APIキー をシークレットマネージャやその他の安全かつアクセス可能な場所に保管してください。

{{% alert %}}
組織スコープのサービスアカウントは、組織内のすべてのチームが所有する非制限プロジェクトにアクセスできる一方で、デフォルトチームの設定が必須です。これは、モデルトレーニングや生成系 AI アプリの環境で `WANDB_ENTITY` 変数が設定されていない場合に、ワークロードが失敗するのを防ぐためです。別のチームのプロジェクトで組織スコープのサービスアカウントを利用したい場合は、`WANDB_ENTITY` 環境変数にそのチームを指定してください。
{{% /alert %}}

## チームスコープのサービスアカウント

チームスコープのサービスアカウントは、そのチーム内のすべてのプロジェクトに対して読み書きが可能です（ただし、そのチームの [制限付きプロジェクト]({{< relref "../access-management/restricted-projects.md#visibility-scopes" >}}) にはアクセスできません）。チームスコープのサービスアカウントが制限付きプロジェクトへアクセスする場合は、そのプロジェクトの管理者が明示的にサービスアカウントを追加する必要があります。

チーム管理者は、 `<WANDB_HOST_URL>/<your-team-name>/service-accounts` でチームスコープのサービスアカウントの APIキー を取得できます。もしくは、チームの **Team settings** から **Service Accounts** タブを参照してください。

新しくチームスコープのサービスアカウントを作成するには：

* チームの **Service Accounts** タブで **New service account** ボタンをクリックします。
* **Name** を入力します。
* 認証方式として **Generate API key (Built-in)** を選択します。
* **Create** をクリックします。
* 新しく作成されたサービスアカウントの横で **Copy API key** をクリックします。
* コピーした APIキー をシークレットマネージャや安全な場所に保管してください。

チームスコープのサービスアカウントを使うモデルのトレーニングや生成系 AI アプリの環境でチームが設定されていない場合、Run や Weaveトレースはそのサービスアカウントの親チームに属するプロジェクトにログされます。この場合、`WANDB_USERNAME` や `WANDB_USER_EMAIL` の変数によるユーザー帰属は、参照されるユーザーがサービスアカウントの親チームのメンバーで _ない限り、機能しません_。

{{% alert color="warning" %}}
チームスコープのサービスアカウントは、自身の親チームと異なる [チームまたは制限付きスコープのプロジェクト]({{< relref "../access-management/restricted-projects.md#visibility-scopes" >}}) に run をログすることはできませんが、他チームの “公開範囲” プロジェクトには run をログすることができます。
{{% /alert %}}

### 外部サービスアカウント

**Built-in** サービスアカウントに加えて、W&B では [アイデンティティ連携]({{< relref "./identity_federation.md#external-service-accounts" >}}) されたIDプロバイダー（IdP）から発行される JSON Web Token (JWT) を活用することで、W&B SDK や CLI でチームスコープの **External service accounts** もサポートしています。