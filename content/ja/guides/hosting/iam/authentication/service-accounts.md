---
title: ワークフローを自動化するためにサービスアカウントを使用する
description: 組織およびチームスコープのサービスアカウントを使用して、自動または非対話型のワークフローを管理する
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-service-accounts
---

サービスアカウントは、チーム内のプロジェクト全体または複数チームにわたって、一般的なタスクを自動で実行できる人間でない（または機械の）ユーザーを表します。

- 組織の管理者は、組織のスコープでサービスアカウントを作成することができます。
- チームの管理者は、そのチームのスコープでサービスアカウントを作成することができます。

サービスアカウントの APIキー により、呼び出し元はサービスアカウントのスコープ内のプロジェクトを読み書きできます。

サービスアカウントは、W&B Modelsの実験管理を自動化したり、W&B Weaveのトレースをログ記録したりするために、複数のユーザーやチームによるワークフローを集中管理することを可能にします。また、`WANDB_USERNAME`または`WANDB_USER_EMAIL`の[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を使用することにより、サービスアカウントで管理されているワークフローに人間ユーザーのアイデンティティを関連付けるオプションもあります。

{{% alert %}}
サービスアカウントは [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})、エンタープライズライセンスのある [セルフマネージド・インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})、および [SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) のエンタープライズアカウントで利用可能です。
{{% /alert %}}

## 組織スコープのサービスアカウント

組織スコープのサービスアカウントは、チームに関係なく、組織内のすべてのプロジェクトを読み書きする権限を持ちます。ただし、[制限付きプロジェクト]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}})は例外です。制限付きプロジェクトにアクセスする前に、そのプロジェクトの管理者は明示的にサービスアカウントをプロジェクトに追加する必要があります。

組織管理者は、組織またはアカウントダッシュボードの **Service Accounts** タブから組織スコープのサービスアカウントの APIキー を取得できます。

新しい組織スコープのサービスアカウントを作成するには：

* 組織ダッシュボードの **Service Accounts** タブで **New service account** ボタンをクリックします。
* **Name** を入力します。
* サービスアカウントのデフォルトチームを選択します。
* **Create** をクリックします。
* 新しく作成されたサービスアカウントの横で **Copy API key** をクリックします。
* コピーした APIキー を秘密管理マネージャーまたは他の安全でアクセス可能な場所に保存します。

{{% alert %}}
組織スコープのサービスアカウントはデフォルトのチームが必要ですが、それでも組織内のすべてのチームが所有する非制限プロジェクトにアクセスできます。これは、 `WANDB_ENTITY` 変数が モデルトレーニング や生成AIアプリの環境に設定されていない場合に、ワークロードが失敗するのを防ぐのに役立ちます。異なるチームのプロジェクトに組織スコープのサービスアカウントを使用するには、そのチームに `WANDB_ENTITY` 環境変数を設定する必要があります。
{{% /alert %}}

## チームスコープのサービスアカウント

チームスコープのサービスアカウントは、そのチーム内のすべてのプロジェクトを読み書きできますが、そのチーム内の[制限付きプロジェクト]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}})は除きます。制限付きプロジェクトにアクセスする前に、そのプロジェクトの管理者は明示的にサービスアカウントをプロジェクトに追加する必要があります。

チームの管理者として、 `<WANDB_HOST_URL>/<your-team-name>/service-accounts` でチームスコープのサービスアカウントの APIキー を取得できます。あるいは、チームの **Team settings** で **Service Accounts** タブを参照してください。

チーム用の新しいチームスコープのサービスアカウントを作成するには：

* チームの **Service Accounts** タブで **New service account** ボタンをクリックします。
* **Name** を入力します。
* 認証メソッドとして **Generate API key (Built-in)** を選択します。
* **Create** をクリックします。
* 新しく作成されたサービスアカウントの横で **Copy API key** をクリックします。
* コピーした APIキー を秘密管理マネージャーまたは他の安全でアクセス可能な場所に保存します。

チームスコープのサービスアカウントを使用する モデルトレーニング や生成AIアプリの環境でチームを設定しないと、モデルのrunやweaveトレースがサービスアカウントの親チーム内の指定されたプロジェクトにログ記録されます。このようなシナリオでは、参照されているユーザーがサービスアカウントの親チームの一部でない限り、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 変数を使用したユーザー帰属は _機能しません_。

{{% alert color="warning" %}}
チームスコープのサービスアカウントは、親チームとは異なるチーム内の [チームスコープか制限スコープのプロジェクト]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ja" >}}) に runをログ記録することはできませんが、他のチーム内の公開範囲プロジェクトには runをログ記録できます。
{{% /alert %}}

### 外部サービスアカウント

**Built-in** サービスアカウントに加えて、W&B は [アイデンティティのフェデレーション]({{< relref path="./identity_federation.md#external-service-accounts" lang="ja" >}}) を使用して、JSON Web Tokens (JWTs) を発行できるアイデンティティプロバイダー (IdPs) とともに、W&B SDK と CLI を用いたチームスコープの **外部サービスアカウント** もサポートしています。