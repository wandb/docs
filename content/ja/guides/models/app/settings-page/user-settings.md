---
title: ユーザー設定を管理する
description: ユーザー設定で、プロフィール情報、アカウントのデフォルト設定、アラート、ベータ製品への参加、GitHub インテグレーション、ストレージ使用状況、アカウントの有効化、Teams
  の作成を管理できます。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-user-settings
    parent: settings
weight: 10
---

ユーザー プロフィールページに移動し、右上のユーザーアイコンを選択します。ドロップダウンから **設定** を選んでください。

## プロフィール

**プロフィール** セクションでは、アカウント名や所属機関の管理・変更ができます。オプションで自己紹介、所在地、個人や所属機関のウェブサイトへのリンク、プロフィール画像のアップロードも可能です。

## イントロダクションの編集

イントロダクションを編集するには、プロフィールの上部にある **編集** をクリックします。表示される WYSIWYG エディターでは Markdown をサポートしています。
1. 行を編集するには、その行をクリックしてください。時間を短縮したい場合は `/` と入力し、リストから Markdown を選べます。
1. 項目のドラッグハンドルを使って、項目の順序を移動できます。
1. ブロックを削除したい場合はドラッグハンドルをクリックし、**削除** をクリックします。
1. 変更を保存するには **保存** をクリックしてください。

### ソーシャルバッジの追加

`@weights_biases` アカウントのフォローバッジを X (旧Twitter) で追加するには、Markdown 形式で HTML の `<img>` タグを使いバッジ画像へのリンクを挿入できます:

```markdown
[![X: @weights_biases](https://img.shields.io/twitter/follow/weights_biases?style=social)](https://x.com/intent/follow?screen_name=weights_biases)
```
`<img>` タグでは `width` や `height` を指定することができます。どちらか一方を指定した場合でも画像の比率は維持されます。

## Teams

**Team** セクションで新しいチームを作成できます。新規チームを作成するには **New team** ボタンを選択し、下記の情報を入力します：

* **Team name** - チーム名です。チーム名は一意である必要があります。チーム名は後から変更できません。
* **Team type** - **Work** または **Academic** ボタンのいずれかを選択します。
* **Company/Organization** - チームに紐づく会社または組織名を入力します。ドロップダウンメニューから既存の会社や組織を選ぶこともできます。新しい組織を登録することも可能です。

{{% alert %}}
チームの作成は管理者アカウントのみ可能です。
{{% /alert %}}

## Beta features

**Beta Features** セクションでは、開発中の新しいプロダクトをいち早く試したり、楽しいアドオン機能を有効化できます。有効にしたいベータ機能の横にあるトグルをオンにしてください。

## アラート

run がクラッシュ・完了した際、またはカスタムアラートを設定した場合に通知を受け取ることができます（[wandb.Run.alert()]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}}) を参照）。通知はメールや Slack で受け取れます。希望するイベントタイプの横でトグルをオンにしてください。

* **Runs finished**: Weights and Biases の run が正常に完了した場合の通知
* **Run crashed**: run が途中で失敗した場合の通知

アラートの設定や管理に関する詳細は [wandb.Run.alert() でアラートを送信する方法]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}}) をご覧ください。

## 個人 GitHub インテグレーション

個人の GitHub アカウントを接続できます。GitHub アカウントを連携するには：

1. **Connect Github** ボタンをクリックします。OAuth 認証ページへリダイレクトされます。
2. **Organization access** セクションでアクセスを許可する組織を選択します。
3. **Authorize** **wandb** を選択します。

## アカウントの削除

アカウントを削除するには、**Delete Account** ボタンをクリックしてください。

{{% alert color="secondary" %}}
アカウントの削除は取り消しできません。
{{% /alert %}}

## ストレージ

**Storage** セクションでは、ご自身のアカウントが Weights and Biases サーバー上で消費している合計メモリ量を確認できます。デフォルトのストレージプランは 100GB です。ストレージと価格の詳細については [Pricing](https://wandb.ai/site/pricing) ページをご覧ください。