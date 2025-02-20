---
title: Manage user settings
description: プロフィール情報、アカウントのデフォルト、アラート、ベータ製品への参加、GitHub インテグレーション、ストレージ使用量、アカウントのアクティベーションを管理し、ユーザー設定で
  Teams を作成します。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-user-settings
    parent: settings
weight: 10
---

ナビゲート先: ユーザープロファイルページ。右上のユーザーアイコンを選択します。ドロップダウンから**設定**を選びます。

## プロフィール

**プロフィール**セクションでは、アカウント名と機関を管理および変更できます。また、バイオグラフィ、所在地、個人または機関のウェブサイトへのリンクを追加し、プロフィール画像をアップロードすることもできます。

## Teams

**Team**セクションで新しいチームを作成します。新しいチームを作成するには、**New team**ボタンを選択し、以下の情報を提供します：

* **Team name** - チームの名前。チーム名は一意でなければなりません。チーム名は変更できません。
* **Team type** - **Work** または **Academic** ボタンを選びます。
* **Company/Organization** - チームの会社または組織の名前を提供します。ドロップダウンメニューから会社または組織を選択します。新しい組織を提供することもできます。

{{% alert %}}
チームを作成できるのは管理アカウントのみです。
{{% /alert %}}

## ベータ機能

**ベータ機能**セクションでは、開発中の新製品のプレビュー機能や追加機能を有効にできます。有効にしたいベータ機能の横にあるトグルスイッチを選択します。

## アラート

[wandb.alert()]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}}) を使って、runがクラッシュしたとき、終了したとき、またはカスタムアラートを設定したときに通知を受け取ります。通知はEmailまたはSlackを通じて受け取ります。通知を受け取りたいイベントタイプの横にあるスイッチを切り替えます。

* **Runs finished**: Weights and Biases runの成功終了かどうか。
* **Run crashed**: runが中断されて終了した場合の通知。

アラートのセットアップと管理の詳細については、[Send alerts with wandb.alert]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}})を参照してください。

## 個人用GitHubインテグレーション

個人用Githubアカウントを接続します。Githubアカウントを接続するには：

1. **Connect Github**ボタンを選択します。これにより、オープン認証（OAuth）ページにリダイレクトされます。
2. **Organization access**セクションでアクセスを許可する組織を選びます。
3. **Authorize** **wandb** を選択します。

## アカウントの削除

**Delete Account** ボタンを選択してアカウントを削除します。

{{% alert color="secondary" %}}
アカウント削除は元に戻せません。
{{% /alert %}}

## ストレージ

**Storage**セクションは、Weights and Biases サーバー上で消費されたアカウントの総メモリ使用量を説明します。デフォルトのストレージプランは100GBです。ストレージと価格設定の詳細については、[Pricing](https://wandb.ai/site/pricing)ページをご覧ください。