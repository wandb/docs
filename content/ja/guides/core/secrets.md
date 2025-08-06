---
title: シークレット
description: W&B シークレットの概要、その仕組み、および開始方法について説明します。
menu:
  default:
    identifier: ja-guides-core-secrets
    parent: core
url: guides/secrets
weight: 1
---

W&B Secret Manager は、アクセストークンやベアラートークン、APIキー、パスワードなどの機密性の高い文字列である _secrets_ を安全かつ一元的に保存、管理、注入できる機能です。Secret Manager を使えば、コードや webhook のヘッダー、[ペイロード]({{< relref path="/guides/core/automations/" lang="ja" >}}) の設定時に機密文字列を直接記述する必要がなくなります。

Secrets は各チームの Secret Manager 内、[チーム設定]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ja" >}}) の **Team secrets** セクションで管理されます。

{{% alert %}}
* Secret の作成・編集・削除は、W&B 管理者のみが行えます。
* Secrets は W&B のコア機能の一部であり、Azure, GCP, AWS にホストした [W&B Server デプロイメント]({{< relref path="/guides/hosting/" lang="ja" >}}) でも利用可能です。別のデプロイメントタイプをご利用の場合は、W&B アカウントチームにご相談ください。
* W&B Server では、ご自身のセキュリティニーズを満たすためのセキュリティ対策の設定はユーザーの責任となります。

  - W&B では、AWS, GCP, Azure が提供する secrets manager の W&B インスタンスに secrets を保存し、高度なセキュリティ機能を確実に設定することを強く推奨します。

  - クラウドの secrets manager (AWS, GCP, Azure) の W&B インスタンスが利用できず、かつクラスター利用時のセキュリティリスクを十分理解している場合を除き、シークレットストアのバックエンドとして Kubernetes クラスターを使用することは推奨しません。
{{% /alert %}}

## シークレットを追加する
シークレットを追加する手順:

1. 受信サービスが Webhook の認証に必要とする場合は、必要なトークンまたは APIキーを生成します。必要であれば、生成した機密文字列はパスワードマネージャー等で安全に保存してください。
1. W&B にログインし、チームの **Settings** ページを開きます。
1. **Team Secrets** セクションで **New secret** をクリックします。
1. 英字・数字・アンダースコア（ `_` ）を使ってシークレット名を入力します。
1. 機密文字列を **Secret** フィールドに貼り付けます。
1. **Add secret** をクリックします。

Webhook の設定時に使用したいシークレットを指定してください。詳細は [Webhook を設定する]({{< relref path="#configure-a-webhook" lang="ja" >}}) セクションをご覧ください。

{{% alert %}}
シークレットを作成すると、[webhook オートメーションのペイロード]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}}) 内で `${SECRET_NAME}` の形式でそのシークレットにアクセスできます。
{{% /alert %}}

## シークレットのローテーション
シークレットをローテーションして値を更新する手順:

1. シークレット行の鉛筆アイコンをクリックし、シークレットの詳細を開きます。
1. **Secret** に新しい値を入力します。必要に応じて **Reveal secret** をクリックして新しい値を確認できます。
1. **Add secret** をクリックします。これでシークレットの値が新しいものに更新され、前の値は参照できなくなります。

{{% alert %}}
シークレットが作成または更新された後は、現在の値を表示することはできません。値の確認や変更が必要な場合は、新しい値でローテーションしてください。
{{% /alert %}}

## シークレットを削除する
シークレットを削除する手順:

1. シークレット行のゴミ箱アイコンをクリックします。
1. 確認ダイアログの内容を読み、**Delete** をクリックします。シークレットは直ちに、かつ完全に削除されます。

## シークレットへのアクセス管理
チームのオートメーションは、チームの secrets を利用できます。シークレットを削除する前に、それを利用しているオートメーションの更新や削除を行い、意図しない動作停止を防いでください。