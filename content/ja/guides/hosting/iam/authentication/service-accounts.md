---
title: サービスアカウントを使用してワークフローを自動化する
description: 組織やチーム単位のサービスアカウントを使って、自動化や非対話型ワークフローを管理する
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-service-accounts
---

サービスアカウントとは、人間以外またはマシンによるユーザーを表し、チーム内や複数チームにわたってプロジェクトに対する一般的なタスクを自動的に実行できるアカウントです。

- 組織管理者は、組織単位でサービスアカウントを作成できます。
- チーム管理者は、そのチーム単位でサービスアカウントを作成できます。

サービスアカウントの APIキー を使うことで、そのサービスアカウントのスコープ内の Projects への読み書きが可能になります。

サービスアカウントを使うことで、複数のユーザーやチームでワークフローを一元的に管理し、W&B Models のための実験管理を自動化したり、W&B Weave のトレースをログしたりできます。サービスアカウントによるワークフローに人間ユーザーの身元を紐づけたい場合は、[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` のいずれかを使用できます。

{{% alert %}}
サービスアカウントは [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、エンタープライズライセンス付きの [自己管理インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})、および [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) のエンタープライズアカウントでご利用いただけます。
{{% /alert %}}

## 組織スコープのサービスアカウント

組織スコープのサービスアカウントには、チームに関係なく、その組織内のすべての Projects への読み書き権限があります（[制限付きプロジェクト]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) を除く）。組織スコープのサービスアカウントが制限付きプロジェクトへアクセスするには、そのプロジェクト管理者が明示的にサービスアカウントをプロジェクトに追加する必要があります。

組織管理者は、組織またはアカウントダッシュボードの **Service Accounts** タブから、組織スコープのサービスアカウントの APIキー を取得できます。

新しい組織スコープのサービスアカウントを作成する方法:

* 組織ダッシュボードの **Service Accounts** タブで **New service account** ボタンをクリックします。
* **Name** を入力します。
* サービスアカウントのデフォルトチームを選択します。
* **Create** をクリックします。
* 新しく作成したサービスアカウントの横にある **Copy API key** をクリックします。
* コピーした APIキー をシークレットマネージャーや、セキュアかつアクセス可能な場所に保管します。

{{% alert %}}
組織スコープのサービスアカウントには、アクセス権限のある全チームの非制限 Projects であっても、デフォルトチームの指定が必須です。これは、モデルのトレーニングや生成 AI アプリの環境で `WANDB_ENTITY` 変数が未設定の場合でも、ワークロードの失敗を防ぐためです。別のチームの Project で組織スコープのサービスアカウントを使う場合は、そのチーム名を `WANDB_ENTITY` 環境変数に設定してください。
{{% /alert %}}

## チームスコープのサービスアカウント

チームスコープのサービスアカウントは、そのチーム内のすべての Projects への読み書きができますが、同じチーム内の [制限付きプロジェクト]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) には管理者による明示的な追加が必要です。

チーム管理者は `<WANDB_HOST_URL>/<your-team-name>/service-accounts` で、またはチームの **Team settings** から **Service Accounts** タブを参照して、チームスコープのサービスアカウントの APIキー を取得できます。

自分のチーム用に新しいチームスコープのサービスアカウントを作成するには：

* チームの **Service Accounts** タブで **New service account** ボタンをクリックします。
* **Name** を入力します。
* 認証方法として **Generate API key (Built-in)** を選択します。
* **Create** をクリックします。
* 新しく作成したサービスアカウントの横にある **Copy API key** をクリックします。
* コピーした APIキー をシークレットマネージャーや、セキュアかつアクセス可能な場所に保管します。

チームスコープのサービスアカウントを利用するモデルのトレーニングや生成 AI アプリの環境でチームを設定しない場合、そのサービスアカウントの親チーム内の指定プロジェクトに対して model run や weave trace のログが保存されます。この場合、`WANDB_USERNAME` や `WANDB_USER_EMAIL` 変数によるユーザーの紐づけは、参照されたユーザーがサービスアカウントの親チームに所属していない限り _機能しません_。

{{% alert color="warning" %}}
チームスコープのサービスアカウントは、親チーム以外の [チームまたは制限付きスコープのプロジェクト]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) には run をログできませんが、別のチームにある公開範囲プロジェクト（open visibility project）には run をログできます。
{{% /alert %}}

### 外部サービスアカウント

**Built-in** サービスアカウントに加え、W&B では [アイデンティティフェデレーション]({{< relref path="./identity_federation.md#external-service-accounts" lang="ja" >}}) を利用した W&B SDK および CLI によるチームスコープの **External service accounts** もサポートしています。これには JWT（JSON Web Tokens）を発行できるアイデンティティプロバイダ（IdP）が必要です。