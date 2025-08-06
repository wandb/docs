---
title: シークレット
description: W&B シークレットの概要、仕組み、そして開始方法をご紹介します。
menu:
  default:
    identifier: secrets
    parent: core
url: guides/secrets
weight: 1
---

W&B Secret Manager を使うと、アクセストークンやベアラートークン、APIキー、パスワードなどのセンシティブな文字列（_secrets_）を、安全かつ一元的に保存・管理・注入できます。W&B Secret Manager を利用することで、センシティブな文字列を直接コードに記述したり、Webhook のヘッダーや [payload]({{< relref "/guides/core/automations/" >}}) 設定時に追加する必要がなくなります。

Secrets は各 Team の Secret Manager で管理され、[team settings]({{< relref "/guides/models/app/settings-page/team-settings/" >}}) 内の **Team secrets** セクションで管理します。

{{% alert %}}
* Secret の作成・編集・削除は W&B Admin のみ実行できます。
* Secrets は W&B のコア機能として含まれており、Azure・GCP・AWS 上にホストする [W&B Server deployments]({{< relref "/guides/hosting/" >}}) でも利用できます。その他のデプロイメントタイプでの利用方法については、W&B の担当者にご相談ください。
* W&B Server を利用する場合、ご自身で必要なセキュリティ対策の設定が求められます。

  - W&B では、AWS・GCP・Azure が提供するクラウドプロバイダーの Secrets Manager 上に W&B インスタンスを設定し、より高度なセキュリティ機能を活用することを強く推奨します。

  - クラウド Secrets Manager（AWS、GCP、Azure）の W&B インスタンスが利用できない特別な場合や、クラスター利用時のセキュリティリスク対策に詳しい場合を除き、Secrets ストアのバックエンドとして Kubernetes クラスターの利用はおすすめしません。
{{% /alert %}}

## シークレットを追加する
シークレットを追加するには:

1. 受信サービスが Webhook の認証に必要とする場合は、必要なトークンや APIキー を生成します。必要に応じて、パスワードマネージャーなど安全な方法でセンシティブな文字列を保存してください。
1. W&B にログインし、チームの **Settings** ページに移動します。
1. **Team Secrets** セクションで **New secret** をクリックします。
1. 英字・数字・アンダースコア（`_`）を使ってシークレットの名前を入力します。
1. センシティブな文字列を **Secret** フィールドに貼り付けます。
1. **Add secret** をクリックします。

Webhook の設定時に、利用したいシークレットを指定してください。詳しくは [Configure a webhook]({{< relref "#configure-a-webhook" >}}) セクションをご覧ください。

{{% alert %}}
一度シークレットを作成すると、そのシークレットは [webhook automation's payload]({{< relref "/guides/core/automations/create-automations/webhook.md" >}}) で `${SECRET_NAME}` の形式で利用できます。
{{% /alert %}}

## シークレットのローテーション
シークレットをローテーションし、値を更新するには:
1. 編集したいシークレットの行にある鉛筆アイコンをクリックします。
1. **Secret** を新しい値に設定します。必要に応じて、**Reveal secret** で新しい値を確認できます。
1. **Add secret** をクリックします。シークレットの値が更新され、以前の値は無効になります。

{{% alert %}}
シークレットは作成・更新後、その値を表示できなくなります。値が必要な場合は、新しい値にローテーションしてください。
{{% /alert %}}

## シークレットを削除する
シークレットを削除するには:
1. シークレットの行にあるゴミ箱アイコンをクリックします。
1. 確認ダイアログをよく読み、**Delete** をクリックします。シークレットは即時かつ完全に削除されます。

## シークレットへのアクセス管理
チームの Automations からチームの Secrets を利用できます。シークレットを削除する前に、そのシークレットを利用している Automation を更新または削除し、意図しない動作停止を防いでください。