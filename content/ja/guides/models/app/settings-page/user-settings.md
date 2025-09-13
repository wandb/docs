---
title: ユーザー設定を管理する
description: ユーザー設定で、プロフィール情報、アカウントのデフォルト、アラート、ベータ版プロダクトへの参加、GitHub インテグレーション、ストレージ使用量、アカウントの有効化を管理し、Teams
  を作成できます。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-user-settings
    parent: settings
weight: 10
---

ユーザー プロファイル ページに移動し、右上のユーザー アイコンを選択します。ドロップダウンから **Settings** を選びます。

## プロフィール

**Profile** セクションでは、アカウント名や所属を管理・変更できます。任意で自己紹介文、所在地、個人または所属先のウェブサイトへのリンク、プロフィール画像のアップロードを追加できます。

## イントロダクションを編集

イントロダクションを編集するには、プロフィール上部の **Edit** をクリックします。開いた WYSIWYG エディタは Markdown をサポートしています。
1. 行を編集するには、その行をクリックします。手早く操作したい場合は `/` と入力し、リストから Markdown を選びます。
1. アイテムのドラッグ ハンドルを使って移動できます。
1. ブロックを削除するには、ドラッグ ハンドルをクリックしてから **Delete** をクリックします。
1. 変更を保存するには **Save** をクリックします。

### ソーシャル バッジを追加

X の `@weights_biases` アカウントのフォローバッジを追加するには、バッジ画像を指す HTML の `<img>` タグを含む、Markdown 形式のリンクを追加できます。

```markdown
[![X: @weights_biases](https://img.shields.io/twitter/follow/weights_biases?style=social)](https://x.com/intent/follow?screen_name=weights_biases)
```
`<img>` タグでは `width`、`height`、またはその両方を指定できます。どちらか一方のみを指定した場合は、画像の縦横比は維持されます。

## Default team
複数の team のメンバーである場合、**Default team** セクションで、run や Weave のトレースで team が指定されていないときに使用する既定の team を設定できます。1 つの team のみのメンバーである場合、その team が既定となり、このセクションは表示されません。

続行するにはタブを選択してください。

{{< tabpane text=true >}}
{{% tab header="Multi-tenant Cloud" %}}
**Default location to create new projects in** の横にあるドロップダウンをクリックし、既定の team を選択します。
{{% /tab %}}
{{% tab header="Dedicated Cloud / Self-Managed" %}}
1. **Default location to create new projects in** の横にあるドロップダウンをクリックし、既定の team または個人の entity を選択します。
1. （任意）管理者が **Account** > **Settings** > **Privacy** で public projects を有効にしている場合は、新しい projects の既定の “公開範囲” を設定します。**Default project privacy in your personal account** の横にあるボタンをクリックし、**Private**（既定）または **Public** を選択します。
1. （任意）管理者が **Account** > **Settings** > **Privacy** で[コードの既定の保存と差分]({{< relref path="/guides/models/app/features/panels/code.md" lang="ja" >}})を有効にしている場合、run でも有効にするには **Enable code saving in your personal account** をクリックします。
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
自動化された 環境 でスクリプトを実行する際に既定の team を指定するには、`WANDB_ENTITY` [環境 変数]({{< relref path="https://docs.wandb.ai/guides/models/track/environment-variables.md" lang="ja" >}})で既定の場所を指定できます。
{{% /alert %}}

## Teams
**Teams** セクションには、あなたのすべての team が一覧表示されます。

1. team 名をクリックすると、その team のページに移動します。
1. 追加の team に参加する権限がある場合は、**We found teams for you to join** の横にある **View teams** をクリックします。
1. 必要に応じて **Hide teams in public profile** をオンにします。

{{% alert %}}
team の作成や管理については、[Teams を管理する]({{< relref path="/guides/models/app/settings-page/teams/" lang="ja" >}})を参照してください。
{{% /alert %}}

## ベータ機能

**Beta Features** セクションでは、開発中の新機能のプレビューや便利なアドオンを任意で有効化できます。有効にしたいベータ機能のトグル スイッチを選択します。

## アラート

run がクラッシュしたとき、完了したとき、または [wandb.Run.alert()]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}}) でカスタム アラートを設定したときに通知を受け取ります。通知は Email または Slack で受信できます。受け取りたいイベントの横にあるスイッチを切り替えてください。

* **Runs finished**: Weights and Biases の run が正常に完了したかどうか。
* **Run crashed**: run が完了せずに失敗した場合の通知。

アラートの設定と管理の詳細は、[wandb.Run.alert() でアラートを送信]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}})を参照してください。

## 個人 GitHub インテグレーション

個人の GitHub アカウントを接続します。接続手順:

1. **Connect Github** ボタンを選択します。OAuth（オープン認可）のページにリダイレクトされます。
2. **Organization access** セクションで、アクセスを許可する組織を選択します。
3. **Authorize** **wandb** を選択します。

## アカウントを削除する

アカウントを削除するには **Delete Account** ボタンを選択します。

{{% alert color="secondary" %}}
アカウントの削除は元に戻せません。
{{% /alert %}}

## ストレージ

**Storage** セクションには、あなたのアカウントが Weights and Biases の サーバー 上で消費した合計ストレージ使用量が表示されます。既定のストレージ プランは 100GB です。ストレージと料金の詳細は、[料金](https://wandb.ai/site/pricing) ページをご覧ください。