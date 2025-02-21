---
title: Manage teams
description: 同僚と協力して、結果を共有し、チーム全体のすべての 実験 を追跡します。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-teams
    parent: settings
weight: 50
---

W&B Teams を利用して、ML チームがより優れたモデルをより速く構築できるような、一元的なワークスペースとして活用しましょう。

* チームが試したすべての experiments を追跡し、作業の重複をなくします。
* 以前にトレーニングしたモデルを保存して再現します。
* 上司や共同研究者と進捗状況や結果を共有します。
* 性能低下時にリグレッションを検知し、即座にアラートを受け取ります。
* モデルの性能を評価し、モデルのバージョンを比較します。

{{< img src="/images/app_ui/teams_overview.webp" alt="" >}}

## コラボレーション Team の作成

1. 無料の W&B アカウントに[**サインアップまたはログイン**](https://app.wandb.ai/login?signup=true)します。
2. ナビゲーションバーの [**Team を招待**] をクリックします。
3. Team を作成し、共同研究者を招待します。

{{% alert %}}
**注記**: 新しい Team を作成できるのは、組織の管理者のみです。
{{% /alert %}}

## Team プロフィールの作成

Team のプロフィールページをカスタマイズして、イントロダクションを表示したり、一般公開または Team メンバーに公開されている reports と Projects を紹介したりできます。Reports、Projects、および外部リンクを表示します。

* 最高の公開 reports を紹介して、訪問者に最高の research をアピールします
* 最もアクティブな Projects を紹介して、チームメイトが Projects を見つけやすくします
* 会社や research ラボの Web サイト、および公開した論文への外部リンクを追加して、共同研究者を見つけます

## Team メンバーの削除

Team 管理者は、Team の設定ページを開き、退会するメンバーの名前の横にある削除ボタンをクリックできます。ユーザーが退会した後も、Team に記録されたすべての runs は残ります。

## Team の役割と権限の管理

同僚を Team に招待するときは、Team の役割を選択します。Team の役割には、次のオプションがあります。

- **管理者**: Team 管理者は、他の管理者または Team メンバーを追加および削除できます。すべての Projects を変更する権限と、完全な削除権限を持っています。これには、runs、Projects、Artifacts、および Sweeps の削除が含まれますが、これらに限定されません。
- **メンバー**: Team の通常のメンバー。管理者は、Team メンバーをメールで招待します。Team メンバーは、他のメンバーを招待できません。Team メンバーは、そのメンバーが作成した runs と Sweep runs のみを削除できます。A と B の 2 人のメンバーがいるとします。メンバー B が Run を Team B の Project からメンバー A が所有する別の Project に移動します。メンバー A は、メンバー B がメンバー A の Project に移動した Run を削除できません。Run を作成したメンバー、または Team 管理者のみが Run を削除できます。
- **閲覧のみ (エンタープライズ限定機能)**: 閲覧のみのメンバーは、runs、Reports、Workspace など、Team 内のアセットを表示できます。Reports をフォローしてコメントできますが、Project の概要、Reports、または runs を作成、編集、または削除することはできません。
- **カスタムロール (エンタープライズ限定機能)**: カスタムロールを使用すると、組織管理者は、**閲覧のみ** または **メンバー** のロールのいずれかに基づいて、追加の権限とともに新しいロールを作成し、きめ細かいアクセス制御を実現できます。次に、Team 管理者は、それらのカスタムロールをそれぞれの Team のユーザーに割り当てることができます。詳細については、[W&B Teams 向けのカスタムロールの導入](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3)を参照してください。
- **サービスアカウント (エンタープライズ限定機能)**: [サービスアカウントを使用してワークフローを自動化する]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md" lang="ja" >}})を参照してください。

{{% alert %}}
W&B では、Team に複数の管理者を設定することをお勧めします。主要な管理者が不在の場合でも、管理操作を継続できるようにすることがベストプラクティスです。
{{% /alert %}}

### Team の設定

Team の設定では、Team とそのメンバーの設定を管理できます。これらの権限を使用すると、W&B 内で Team を効果的に監督および編成できます。

| 権限                  | 閲覧のみ | Team メンバー | Team 管理者 |
| --------------------- | -------- | ------------- | ----------- |
| Team メンバーの追加     |          |               | X           |
| Team メンバーの削除     |          |               | X           |
| Team の設定の管理       |          |               | X           |

### モデルレジストリ

次の表に、特定の Team のすべての Projects に適用される権限を示します。

| 権限                       | 閲覧のみ | Team メンバー | モデルレジストリ管理者 | Team 管理者 |
| -------------------------- | -------- | ------------- | ------------------ | ----------- |
| エイリアスの追加               |          | X             | X                  | X           |
| レジストリへのモデルの追加       |          | X             | X                  | X           |
| レジストリ内のモデルの表示       | X        | X             | X                  | X           |
| モデルのダウンロード            | X        | X             | X                  | X           |
| レジストリ管理者の追加/削除 |          |               | X                  | X           |
| 保護されたエイリアスの追加/削除 |          |               | X                  |             |

保護されたエイリアスの詳細については、[モデルレジストリ]({{< relref path="/guides/models/registry/model_registry/access_controls.md" lang="ja" >}})のチャプターを参照してください。

### Reports

Report 権限は、Reports の作成、表示、および編集へのアクセスを許可します。次の表に、特定の Team のすべての Reports に適用される権限を示します。

| 権限         | 閲覧のみ | Team メンバー                                  | Team 管理者 |
| ------------ | -------- | -------------------------------------------- | ----------- |
| Reports の表示 | X        | X                                            | X           |
| Reports の作成 |          | X                                            | X           |
| Reports の編集 |          | X (Team メンバーは自分の Reports のみ編集可能) | X           |
| Reports の削除 |          | X (Team メンバーは自分の Reports のみ編集可能) | X           |

### Experiments

次の表に、特定の Team のすべての experiments に適用される権限を示します。

| 権限                                                                              | 閲覧のみ | Team メンバー | Team 管理者 |
| ----------------------------------------------------------------------------------- | -------- | ------------- | ----------- |
| Experiment のメタデータの表示 (履歴メトリクス、システムメトリクス、ファイル、およびログを含む) | X        | X             | X           |
| Experiment パネルとワークスペースの編集                                                   |          | X             | X           |
| Experiment のログ                                                                     |          | X             | X           |
| Experiment の削除                                                                   |          | X (Team メンバーは自分が作成した experiments のみ削除可能) | X           |
| Experiment の停止                                                                   |          | X (Team メンバーは自分が作成した experiments のみ停止可能)   | X           |

### Artifacts

次の表に、特定の Team のすべての Artifacts に適用される権限を示します。

| 権限             | 閲覧のみ | Team メンバー | Team 管理者 |
| ---------------- | -------- | ------------- | ----------- |
| Artifacts の表示  | X        | X             | X           |
| Artifacts の作成  |          | X             | X           |
| Artifacts の削除  |          | X             | X           |
| メタデータの編集    |          | X             | X           |
| エイリアスの編集   |          | X             | X           |
| エイリアスの削除   |          | X             | X           |
| Artifact のダウンロード |          | X             | X           |

### システム設定 (W&B Server のみ)

システム権限を使用して、Teams とそのメンバーを作成および管理し、システム設定を調整します。これらの権限を使用すると、W&B インスタンスを効果的に管理および保守できます。

| 権限                     | 閲覧のみ | Team メンバー | Team 管理者 | システム管理者 |
| ------------------------ | -------- | ------------- | ----------- | -------------- |
| システム設定の構成           |          |               |             | X              |
| Teams の作成/削除          |          |               |             | X              |

### Team サービスアカウントの振る舞い

* トレーニング環境で Team を構成する場合、その Team のサービスアカウントを使用して、その Team 内のプライベートまたはパブリック Projects のいずれかで runs をログに記録できます。さらに、**WANDB_USERNAME** または **WANDB_USER_EMAIL** 変数が環境に存在し、参照されているユーザーがその Team の一部である場合は、それらの runs をユーザーに属性付けできます。
* トレーニング環境で Team を構成**しない**で、サービスアカウントを使用する場合、runs はそのサービスアカウントの親 Team 内の名前付き Project にログに記録されます。この場合も、**WANDB_USERNAME** または **WANDB_USER_EMAIL** 変数が環境に存在し、参照されているユーザーがサービスアカウントの親 Team の一部である場合は、runs をユーザーに属性付けできます。
* サービスアカウントは、親 Team とは異なる Team のプライベート Project に runs をログに記録できません。サービスアカウントは、Project の可視性が「オープン」に設定されている場合にのみ、Project に runs をログに記録できます。

#### イントロにソーシャルバッジを追加する

イントロで `/` を入力し、Markdown を選択して、バッジを表示する Markdown スニペットを貼り付けます。WYSIWYG に変換すると、サイズを変更できます。

たとえば、Twitter のフォローバッジを追加するには、`[{{< img src="https://img.shields.io/twitter/follow/weights_biases?style=social" alt="Twitter: @weights_biase" >}}](https://twitter.com/intent/follow?screen_name=weights_biases` の `weights_biases` を Twitter のユーザー名に置き換えます。

[{{< img src="https://img.shields.io/twitter/follow/weights_biases?style=social" alt="Twitter: @weights_biases" >}}](https://twitter.com/intent/follow?screen_name=weights_biases)

## Team トライアル

W&B のプランの詳細については、[料金ページ](https://wandb.ai/site/pricing)を参照してください。ダッシュボード UI または [Export API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使用して、いつでもすべてのデータをダウンロードできます。

## プライバシー設定

Team のすべての Projects のプライバシー設定は、Team の設定ページで確認できます。
`app.wandb.ai/teams/your-team-name`

## 高度な設定

### セキュアストレージコネクタ

Team レベルのセキュアストレージコネクタを使用すると、Teams は W&B で独自のクラウドストレージバケットを使用できます。これにより、非常に機密性の高いデータまたは厳格なコンプライアンス要件を持つ Teams に対して、より優れたデータアクセス制御とデータ分離が提供されます。詳細については、[セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を参照してください。
