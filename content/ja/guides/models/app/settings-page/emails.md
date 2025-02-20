---
title: Manage email settings
description: 設定ページからメールを管理します。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-emails
    parent: settings
weight: 40
---

Add, delete, manage email types and primary email addresses in your W&B Profile Settings page. Select your profile icon in the upper right corner of the W&B dashboard. From the dropdown, select **設定**. Within the Settings page, scroll down to the Emails ダッシュボード:

{{< img src="/images/app_ui/manage_emails.png" alt="" >}}

## プライマリメールの管理

プライマリメールは 😎 絵文字でマークされています。プライマリメールは、W&B アカウントを作成したときに提供したメールで自動的に設定されます。

Weights & Biases アカウントに関連付けられたプライマリメールを変更するには、ケバブのドロップダウンを選択します。

{{% alert %}}
確認済みのメールのみがプライマリとして設定できます
{{% /alert %}}

{{< img src="/images/app_ui/primary_email.png" alt="" >}}

## メールの追加

**+ Add Email** を選択してメールを追加します。これにより Auth0 ページに移動します。新しいメールの認証情報を入力するか、シングルサインオン (SSO) を使用して接続できます。

## メールの削除

ケバブのドロップダウンを選択し、**Delete Emails** を選択して、W&B アカウントに登録されているメールを削除します。

{{% alert %}}
プライマリメールは削除できません。削除する前に別のメールをプライマリメールとして設定する必要があります。
{{% /alert %}}

## ログインメソッド

**ログインメソッド**列には、アカウントに関連付けられたログインメソッドが表示されます。

W&B アカウントを作成すると、確認メールがメールアカウントに送信されます。メールアカウントは、メールアドレスを確認するまで未確認とみなされます。未確認のメールは赤で表示されます。

元の確認メールがメールアカウントに送信された場合で、もう一度ログインを試み、確認メールを再取得してください。

アカウントのログインの問題については、support@wandb.com にお問い合わせください。