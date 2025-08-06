---
title: ユーザー設定を管理する
description: ユーザー設定では、プロフィール情報、アカウントのデフォルト設定、アラート、ベータ製品への参加、GitHub インテグレーション、ストレージ使用状況、アカウントの有効化、Teams
  の作成などを管理できます。
menu:
  default:
    identifier: user-settings
    parent: settings
weight: 10
---

ユーザープロフィールページに移動し、右上のユーザーアイコンを選択してください。ドロップダウンメニューから **設定** を選びます。

## プロフィール

**プロフィール** セクションでは、アカウント名や所属機関の管理・変更ができます。任意で略歴、所在地、個人または所属機関のウェブサイトのリンク、プロフィール画像のアップロードも行えます。

## イントロダクションの編集

イントロダクションを編集するには、プロフィールの上部にある **編集** をクリックしてください。開かれる WYSIWYG エディタは Markdown に対応しています。
1. ラインを編集するには、その行をクリックします。効率化のために `/` をタイプし、リストから Markdown を選択することもできます。
1. アイテムのドラッグハンドルで順番を入れ替えることができます。
1. ブロックを削除する場合はドラッグハンドルをクリックし、続いて **削除** をクリックします。
1. 変更を保存するには **保存** をクリックします。

### ソーシャルバッジを追加

X の `@weights_biases` アカウントのフォローバッジを追加するには、バッジ画像への HTML の `<img>` タグを含む Markdown スタイルのリンクを追加できます。

```markdown
[![X: @weights_biases](https://img.shields.io/twitter/follow/weights_biases?style=social)](https://x.com/intent/follow?screen_name=weights_biases)
```
`<img>` タグ内では `width` や `height`、または両方を指定できます。どちらか一方だけを指定した場合、画像の比率は保持されます。

## Teams

**Team** セクションで新しいチームを作成できます。新しいチームを作成するには、**New team** ボタンを選択し、次の情報を入力してください。

* **Team name** - チームの名前です。チーム名は一意である必要があります。チーム名は後から変更できません。
* **Team type** - **Work** または **Academic** ボタンのどちらかを選択します。
* **Company/Organization** - チームが属する会社または組織名を入力します。ドロップダウンメニューから会社または組織を選択してください。新しい組織も入力可能です。

{{% alert %}}
管理者アカウントのみがチームを作成できます。
{{% /alert %}}

## Beta features

**Beta Features** セクションでは、開発中の新製品の先行体験や便利な追加機能を有効にできます。有効化したいベータ機能の横にあるトグルスイッチを選択してください。

## アラート

run のクラッシュ、完了、もしくはカスタムアラートを設定したときに通知を受け取れます。[wandb.Run.alert()]({{< relref "/guides/models/track/runs/alert.md" >}}) をご利用ください。通知はメールや Slack で受け取ることができます。受信したいアラート項目の隣のスイッチを切り替えてください。

* **Runs finished**: Weights and Biases の run が正常完了したかどうか。
* **Run crashed**: run が失敗した場合に通知します。

アラートの設定や管理方法について詳しくは、[Send alerts with wandb.Run.alert()]({{< relref "/guides/models/track/runs/alert.md" >}}) をご覧ください。

## 個人 GitHub インテグレーション

個人の GitHub アカウントを接続できます。接続手順は以下の通りです。

1. **Connect Github** ボタンを選択します。OAuth（オープン認証）ページへリダイレクトされます。
2. **Organization access** セクションで、アクセスを許可する組織を選択します。
3. **Authorize** **wandb** を選択します。

## アカウントの削除

**Delete Account** ボタンを選択すると、アカウントを削除します。

{{% alert color="secondary" %}}
アカウントを削除すると、取り消しはできません。
{{% /alert %}}

## ストレージ

**ストレージ** セクションでは、Weights and Biases サーバー上でアカウントが使用した総メモリ容量を表示します。デフォルトのストレージプランは 100GB です。ストレージや料金についての詳細は [Pricing](https://wandb.ai/site/pricing) ページをご確認ください。