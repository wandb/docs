---
title: チームを管理する
description: 同僚と協力し、結果を共有し、チーム全体の実験をすべて追跡できます。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-teams
    parent: settings
weight: 50
---

W&B Teams を使って、ML チームのための中央ワークスペースを構築し、より良いモデルをより早く作りましょう。

* **チームの全ての Experiments を記録** し、作業の重複を防ぎます。
* **過去にトレーニングした Models を保存・再現** できます。
* **進捗や結果を上司や協力者とシェア** できます。
* **パフォーマンスの低下をすぐにキャッチし通知** を受け取れます。
* **Model の性能をベンチマークし、バージョンごとに比較** できます。

{{< img src="/images/app_ui/teams_overview.webp" alt="Teams workspace overview" >}}

## 協働チームを作成する

1. [新規登録またはログイン](https://app.wandb.ai/login?signup=true) して、無料の W&B アカウントを作成します。
2. ナビゲーションバーから **Invite Team** をクリックします。
3. チームを作成し、協力者を招待しましょう。
4. チームの設定方法は [Manage team settings]({{< relref path="team-settings.md#privacy" lang="ja" >}}) を参照してください。

{{% alert %}}
**注意**: 組織の管理者のみが新しい Team を作成できます。
{{% /alert %}}

## チームプロフィールを作成する

チームのプロフィールページをカスタマイズし、イントロダクションや公開・チームメンバー向けの Reports や Projects を表示できます。Reports や Projects、外部リンクなども掲載可能です。

* **最も優れた研究成果を公開レポートで紹介** し、来訪者にアピール
* **アクティブな Projects をハイライト** し、チームメンバーが見つけやすくします
* **外部リンク追加で共同研究者を発見** ー 会社や研究室のウェブサイトや公開論文などを掲載







## チームメンバーの削除

チーム管理者はチーム設定ページから退会メンバーの横にある削除ボタンをクリックすることで、メンバーを削除できます。チームに記録された Run は、ユーザーが抜けても残ります。

## チームの役割と権限を管理する
チームへ招待する際に、以下から適切なロールを選んでメンバーを追加できます。

- **Admin**: チーム管理者。他の管理者やメンバーの追加・削除が可能です。全ての Projects の編集・削除権限を持ちます（Run、Projects、Artifacts、Sweeps の削除なども含む）。
- **Member**: 通常のメンバー。基本的に Admin のみがメンバー招待可能です。この振る舞いを変更したい場合は [Manage team settings]({{< relref path="team-settings.md#privacy" lang="ja" >}}) をご参照ください。

チームメンバーは自分が作成した Run のみ削除可能です。たとえば、A と B という 2 人がいて、B が Team B の Project から Member A の Project へ Run を移動した場合、A はその Run を削除できません。Admin は全てのメンバーが作成した Run や Sweep Run を管理できます。
- **View-Only (Enterprise限定機能)**: View-Only メンバーはチーム内の Run、Reports、Workspace 等のアセットを閲覧可能です。Reports のフォローやコメントはできますが、Project 概要や Reports、Run の作成・編集・削除はできません。
- **Custom roles (Enterprise限定機能)**: Custom roles では、組織の管理者が **View-Only** または **Member** をベースに、さらなる権限を加えた独自のロールを作成できます。チーム管理者はこれらのカスタムロールをチームユーザーに割り当て可能です。詳しくは [Introducing Custom Roles for W&B Teams](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) をご参照ください。
- **Service accounts (Enterprise限定機能)**: ワークフロー自動化のためのサービスアカウントの利用については [Use service accounts to automate workflows]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md" lang="ja" >}}) をご覧ください。

{{% alert %}}
１つのチームに Admin が複数いることを W&B では推奨しています。これにより、主要な管理者が不在でも管理作業が継続できます。
{{% /alert %}}

### チーム設定
チーム設定では、チームやメンバーに関する設定を管理できます。これらの権限を活用して、W&B 内でチームを効果的に統括しましょう。

| 権限                      | View-Only | Team Member | Team Admin | 
| ------------------------- | --------- | ----------- | ---------- |
| チームメンバー追加         |           |             |     X      |
| チームメンバー削除         |           |             |     X      |
| チーム設定の管理           |           |             |     X      |

### Registry
以下の表は、チーム内すべての Projects に適用される Registry 権限です。

| 権限                                   | View-Only | Team Member | Registry Admin | Team Admin | 
| --------------------------------------- | --------- | ----------- | -------------- | ---------- |
| Alias の追加                            |           | X           | X              | X          |
| Models を Registry へ追加               |           | X           | X              | X          |
| Registry 内の Models 閲覧               | X         | X           | X              | X          |
| Models のダウンロード                   | X         | X           | X              | X          |
| Registry Admin の追加・削除             |           |             | X              | X          | 
| Protected Alias の追加・削除            |           |             | X              |            | 

Protected Alias について詳しくは [Registry Access Controls]({{< relref path="/guides/core/registry/model_registry/access_controls.md" lang="ja" >}}) をご覧ください。

### Reports
Reports の権限は、チーム内すべての Reports への作成・閲覧・編集権限を示します。

| 権限             | View-Only | Team Member                                   | Team Admin | 
| ---------------- | --------- | --------------------------------------------- | ---------- |
| Reports 閲覧     | X         | X                                             | X          |
| Reports 作成     |           | X                                             | X          |
| Reports 編集     |           | X (自分の Reports のみ編集可)                 | X          |
| Reports 削除     |           | X (自分の Reports のみ編集可)                 | X          |

### Experiments
以下の表は、チーム全体の Experiments についての権限です。

| 権限                                                                             | View-Only | Team Member | Team Admin | 
| -------------------------------------------------------------------------------- | --------- | ----------- | ---------- |
| 実験のメタデータ閲覧 (履歴メトリクス・システムメトリクス・ファイル・ログ含む)       | X         | X           | X          |
| 実験パネル・Workspace の編集                                                      |           | X           | X          |
| 実験のログ                                                                        |           | X           | X          |
| 実験の削除                                                                        |           | X (自分が作成した Experiment のみ可) | X |
| 実験の停止                                                                        |           | X (自分が作成した Experiment のみ可) | X |

### Artifacts
以下は、チーム全体の Artifacts 権限です。

| 権限                 | View-Only | Team Member | Team Admin | 
| -------------------- | --------- | ----------- | ---------- |
| Artifacts 閲覧       | X         | X           | X          |
| Artifacts 作成       |           | X           | X          |
| Artifacts 削除       |           | X           | X          |
| メタデータ編集       |           | X           | X          |
| Alias 編集           |           | X           | X          |
| Alias 削除           |           | X           | X          |
| Artifact ダウンロード|           | X           | X          |

### システム設定 (W&B Server のみ)
システム権限を利用して、Teams やメンバーの作成・管理、システム設定の調整ができます。これらの権限で W&B インスタンス自体の管理が行えます。

| 権限                      | View-Only | Team Member | Team Admin | System Admin | 
| ------------------------- | --------- | ----------- | ---------- | ------------ |
| システム設定の管理         |           |             |            | X            |
| Team の作成・削除          |           |             |            | X            |

### チームサービスアカウントの挙動

* トレーニング環境でチームを設定した場合、そのチームのサービスアカウントを使って、プライベートまたはパブリックな Team Projects に Run を記録できます。さらに、 **WANDB_USERNAME** または **WANDB_USER_EMAIL** 変数が環境内に存在し、かつ参照ユーザーがその Team メンバーであれば、その Run にユーザーを紐付けできます。
* トレーニング環境にチームを設定せずサービスアカウントのみを使用する場合は、そのサービスアカウントの親チーム内の指定 Project に Run が記録されます。この場合も **WANDB_USERNAME** または **WANDB_USER_EMAIL** 変数が存在し、かつ参照ユーザーが親チームメンバーであればユーザー割り当てが可能です。
* サービスアカウントは親 Team 以外の Team のプライベート Project には Run をログできません。Project の公開設定が `Open` の場合のみ、その Project へのログが可能です。

## Team トライアル

W&B の各種プラン詳細は [料金ページ](https://wandb.ai/site/pricing) をご覧ください。すべてのデータは、ダッシュボード UI または [Export API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) でいつでもダウンロード可能です。

## プライバシー設定

チーム内全ての Projects のプライバシー設定は、チーム設定ページから確認できます:
`app.wandb.ai/teams/your-team-name`

## 詳細な設定

### セキュアストレージコネクタ

チーム単位のセキュアストレージコネクタを利用することで、W&B でチーム独自のクラウドストレージバケットを使うことができます。これにより、高度なデータアクセス管理や分離が求められる場合でも安全に利用できます。詳細は [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) をご参照ください。