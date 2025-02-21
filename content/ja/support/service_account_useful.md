---
title: What is a service account, and why is it useful?
menu:
  support:
    identifier: ja-support-service_account_useful
tags:
- administrator
toc_hide: true
type: docs
---

サービスアカウント（エンタープライズ限定の機能）は、人間ではない、またはマシン ユーザーを表し、チームや Project 全体で、または特定の人間 ユーザーに固有ではない一般的なタスクを自動化できます。チーム内でサービスアカウントを作成し、その APIキー を使用して、そのチーム内の Project からの読み取りと書き込みを行うことができます。

とりわけ、サービスアカウントは、定期的な再トレーニング、夜間のビルドなど、wandb に記録される自動ジョブの追跡に役立ちます。必要に応じて、[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` を使用して、これらのマシンで起動された run にユーザー名を関連付けることができます。

詳細については、[チームサービスアカウントの振る舞い]({{< relref path="/guides/models/app/settings-page/teams.md#team-service-account-behavior" lang="ja" >}})を参照してください。

チームのサービスアカウントの APIキー は、`<WANDB_HOST_URL>/<your-team-name>/service-accounts` で取得できます。または、チームの [**チーム設定**] に移動し、[**サービスアカウント**] タブを参照することもできます。

チームの新しいサービスアカウントを作成するには:
* チームの [**サービスアカウント**] タブで [**+ 新しいサービスアカウント**] ボタンを押します。
* [**名前**] フィールドに名前を入力します。
* 認証方法として [**APIキー の生成（組み込み）**] を選択します。
* [**作成**] ボタンを押します。
* 新しく作成したサービスアカウントの [**APIキー のコピー**] ボタンをクリックして、秘密鍵マネージャーまたは別の安全でアクセス可能な場所に保存します。

{{% alert %}}
**組み込み** のサービスアカウントとは別に、W&B は [SDK および CLI のアイデンティティフェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" lang="ja" >}}) を使用した **外部サービスアカウント** もサポートしています。JSON Web Tokens（JWT）を発行できるアイデンティティプロバイダーで管理されているサービスIDを使用して W&B タスクを自動化する場合は、外部サービスアカウントを使用してください。
{{% /alert %}}
