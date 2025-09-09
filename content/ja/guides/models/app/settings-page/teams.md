---
title: Teams を管理する
description: 同僚と共同作業し、結果を共有し、チーム全体の実験をすべて追跡しましょう。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-teams
    parent: settings
weight: 50
---

W&B Teams を ML チームの中核となる Workspace として使い、より優れた Models をより速く構築しましょう。
* チームが試したすべての Experiments を追跡し、重複作業を防ぐ。
* 学習済みの models を保存し、再現する。
* 進捗と結果を上司や共同研究者と共有する。
* リグレッションを検知し、性能が低下したら即座にアラートを受け取る。
* model の性能をベンチマークし、model のバージョンを比較する。

{{< img src="/images/app_ui/teams_overview.webp" alt="Teams Workspace の概要" >}}

## 共同作業のためのチームを作成

1. 無料の W&B アカウントに [Sign up or log in](https://app.wandb.ai/login?signup=true) します。
2. ナビゲーションバーで **Invite Team** をクリックします。
3. チームを作成し、共同編集者を招待します。
4. チームを設定するには、[Manage team settings]({{< relref path="team-settings.md#privacy" lang="ja" >}}) を参照してください。

{{% alert %}}
注意: 新しいチームを作成できるのは、組織の管理者のみです。
{{% /alert %}}

## チームプロフィールを作成

チームのプロフィールページをカスタマイズして、紹介文を掲載したり、公開またはチームメンバーに見える Reports や Projects を紹介できます。Reports、Projects、外部リンクを掲載しましょう。
* 公開 Reports を掲載して、最高の研究を訪問者にアピール
* 最もアクティブな Projects を紹介して、チームメイトが見つけやすくする
* 会社や研究室のサイト、発表した論文などへの外部リンクを追加して、共同研究者を見つける







## チームメンバーの削除

チーム管理者はチーム設定ページを開き、退会するメンバー名の横にある削除ボタンをクリックできます。チームにログされた runs は、ユーザーが退出した後も残ります。

## チームのロールと権限を管理

同僚をチームに招待する際に、チームロールを選択します。利用できるロールは次のとおりです。
- **Admin**: チーム管理者は、他の管理者やチームメンバーを追加・削除できます。すべての Projects を変更でき、完全な削除権限を持ちます。これには runs、Projects、Artifacts、Sweeps の削除などが含まれます。
- **Member**: 通常のチームメンバー。デフォルトでは、Admin のみがチームメンバーを招待できます。この振る舞いを変更するには、[Manage team settings]({{< relref path="team-settings.md#privacy" lang="ja" >}}) を参照してください。
- **View-Only (Enterprise-only feature)**: View-Only メンバーは、チーム内の runs、reports、workspaces などのアセットを閲覧できます。Reports をフォローしたりコメントしたりできますが、project overview、Reports、runs の作成・編集・削除はできません。
- **Custom roles (Enterprise-only feature)**: Custom roles により、組織の管理者は **View-Only** または **Member** をベースに、きめ細かなアクセス制御を実現する追加権限を組み合わせて新しいロールを作成できます。チーム管理者は、それらの Custom roles を各自のチーム内のユーザーに割り当てられます。詳しくは [Introducing Custom Roles for W&B Teams](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) を参照してください。

チームメンバーは、自分が作成した runs のみ削除できます。たとえば、メンバー A と B がいるとします。メンバー B が、チーム B の Project からメンバー A が所有する別の Project に run を移動した場合、メンバー A はメンバー B が自分の Project に移動した run を削除できません。Admin は、どのチームメンバーが作成した runs や sweep runs でも管理できます。

### サービスアカウント

ユーザーロールに加えて、自動化のために **サービスアカウント** も利用できます。サービスアカウントは Users ではなく、自動化されたワークフローに用いる非人間の識別子です。詳細は [Use service accounts to automate workflows]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md" lang="ja" >}}) を参照してください。

{{% alert %}}
一次管理者が不在でも管理作業を継続できるよう、チーム内に複数の Admin を割り当てることを W&B は推奨します。
{{% /alert %}}

### チーム設定

Team settings では、チームおよびメンバーの設定を管理できます。これらの権限により、W&B 内でチームを効果的に統括・編成できます。

| Permissions         | View-Only | Team Member | Team Admin | 
| ------------------- | --------- | ----------- | ---------- |
| Add team members    |           |             |     X      |
| Remove team members |           |             |     X      |
| Manage team settings|           |             |     X      |

### Reports

Report permissions により、Reports の作成・閲覧・編集へのアクセスが付与されます。次の表は、特定のチーム全体の Reports に適用される権限を示します。

| Permissions   | View-Only | Team Member                                     | Team Admin | 
| -----------   | --------- | ----------------------------------------------- | ---------- |
|View reports   | X         | X                                               | X          |
|Create reports |           | X                                               | X          |
|Edit reports   |           | X (team members can only edit their own reports)| X          |
|Delete reports |           | X (team members can only edit their own reports)| X          |

### Experiments

次の表は、特定のチーム全体の Experiments に適用される権限を示します。

| Permissions | View-Only | Team Member | Team Admin | 
| ------------------------------------------------------------------------------------ | --------- | ----------- | ---------- |
| View experiment metadata (includes history metrics, system metrics, files, and logs) | X         | X           | X          |
| Edit experiment panels and workspaces                                                |           | X           | X          |
| Log experiments                                                                      |           | X           | X          |
| Delete experiments                                                                   |           | X (team members can only delete experiments they created) |  X  |
|Stop experiments                                                                      |           | X (team members can only stop experiments they created)   |  X  |

### Artifacts

次の表は、特定のチーム全体の Artifacts に適用される権限を示します。

| Permissions      | View-Only | Team Member | Team Admin | 
| ---------------- | --------- | ----------- | ---------- |
| View artifacts   | X         | X           | X          |
| Create artifacts |           | X           | X          |
| Delete artifacts |           | X           | X          |
| Edit metadata    |           | X           | X          |
| Edit aliases     |           | X           | X          |
| Delete aliases   |           | X           | X          |
| Download artifact|           | X           | X          |

### System settings (W&B Server only)

System permissions を使って、Teams とそのメンバーの作成・管理やシステム設定の調整を行います。これらの権限により、W&B インスタンスを効果的に管理・運用できます。

| Permissions              | View-Only | Team Member | Team Admin | System Admin | 
| ------------------------ | --------- | ----------- | ---------- | ------------ |
| Configure system settings|           |             |            | X            |
| Create/delete teams      |           |             |            | X            |

### チーム サービスアカウントの振る舞い

* トレーニング 環境でチームを設定した場合、そのチームの サービスアカウント を使って、そのチーム内の private または public な Projects のいずれにも runs をログできます。さらに、環境に **WANDB_USERNAME** または **WANDB_USER_EMAIL** 変数が存在し、参照されるユーザーがそのチームの一員であれば、その runs をそのユーザーに帰属させることができます。
* トレーニング 環境でチームを設定していない状態で サービスアカウント を使用すると、runs はその サービスアカウント の親チーム内で指定した Project にログされます。この場合でも、環境に **WANDB_USERNAME** または **WANDB_USER_EMAIL** 変数が存在し、参照されるユーザーが サービスアカウント の親チームの一員であれば、その runs をそのユーザーに帰属させることができます。
* サービスアカウント は、親チームとは異なるチームの private Project に runs をログできません。Project の可視性が `Open` に設定されている場合にのみ、その Project に runs をログできます。

## チームのトライアル

W&B の各プランについては [pricing page](https://wandb.ai/site/pricing) を参照してください。ダッシュボード UI から、または [Export API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) を使って、いつでもすべてのデータをダウンロードできます。

## プライバシー設定

チーム設定ページで、すべてのチーム Projects のプライバシー設定を確認できます:
`app.wandb.ai/teams/your-team-name`

## 高度な設定

### Secure storage connector

チーム単位の secure storage connector により、Teams は自分たちのクラウド ストレージ バケットを W&B と組み合わせて利用できます。これは、機密性の高いデータや厳格なコンプライアンス要件を持つ Teams に対して、より高いデータ アクセス制御とデータ分離を提供します。詳しくは [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を参照してください。